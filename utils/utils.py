import os
import re
import torch
import base64
import io
import traceback
import numpy as np
import open3d as o3d
from PIL import Image
from matplotlib import pyplot as plt
from models.FGC_graspnet.utils.data_utils import create_point_cloud_from_depth_image

from utils.config import langsam_actor
from utils.vila_client import reasoning_client

try:
    from utils.state_memory import state_memory
except ImportError:  # pragma: no cover
    from .state_memory import state_memory

REACHABILITY_SYSTEM_PROMPT = (
    "You judge whether the target object in a numbered bin-picking image is graspable now by a parallel gripper.\n"
    "Each visible object already has one object ID drawn on it.\n"
    "Meaning:\n"
    "- free: the target can be grasped directly now.\n"
    "- blocked: another visible object must be removed first.\n"
    "Judge blocked only when another object clearly does at least one of these:\n"
    "- sits on the target,\n"
    "- crosses over the target,\n"
    "- covers the target's main graspable area,\n"
    "- prevents a top-down gripper approach.\n"
    "Do not answer blocked just because another object is nearby.\n"
    "If blocking is unclear, answer free.\n"
    "Output exactly one word: free or blocked."
)

ORIGINAL_SCENE_SYSTEM_PROMPT = (
    "You analyze the original unnumbered scene image for robotic bin picking with a parallel gripper.\n"
    "Decide whether the target can be grasped directly.\n"
    "Use blocked only when another object clearly sits on the target, crosses over it, covers its main graspable area, or prevents a top-down gripper approach.\n"
    "Do not treat nearby objects as blockers by default.\n"
    "If blocking is unclear, prefer free.\n"
    "Output exactly one line in one of these forms:\n"
    "free\n"
    "blocked: <short blocker description>\n"
    "Use a short blocker description with color and category when possible."
)

TARGET_ID_SYSTEM_PROMPT = (
    "You identify the target object in a numbered bin-picking image.\n"
    "Your only task is to return the one visible object ID that best matches the text description.\n"
    "Match object category and color first.\n"
    "Do not return a nearby object, support object, lower object, or blocker unless it is itself the described target.\n"
    "Output exactly one line containing only the object ID."
)

BLOCKER_ID_SYSTEM_PROMPT = (
    "You select one blocker object from a small candidate list in a numbered bin-picking image.\n"
    "The target cannot be grasped directly.\n"
    "A blocker is an object that sits on the target, crosses over it, covers its main graspable area, or prevents a top-down gripper approach.\n"
    "Choose exactly one blocker ID from the provided blocking candidate IDs.\n"
    "Do not output the target object's own ID.\n"
    "Output exactly one line containing only the object ID."
)


def build_candidate_id_text(labeled_text):
    visible_ids = []
    if isinstance(labeled_text, list):
        lines = labeled_text
        id_lines = "\n".join(labeled_text)
    else:
        lines = labeled_text.strip().splitlines()
        id_lines = labeled_text.strip()

    for line in lines[1:]:
        parts = line.split()
        if parts:
            visible_ids.append(parts[0])
    return visible_ids, id_lines


def build_reachability_user_text(task_text, labeled_text):
    visible_ids, id_lines = build_candidate_id_text(labeled_text)
    return (
        "Task: decide whether the target object is graspable now.\n"
        f"Target instruction: {task_text}\n"
        f"Visible candidate IDs: {', '.join(visible_ids)}\n"
        "Numbered objects:\n"
        f"{id_lines}\n"
        "First identify the described target.\n"
        "Then judge whether a parallel gripper can approach it from above now.\n"
        "Answer blocked only if another visible object clearly sits on it, crosses over it, covers its main graspable area, or prevents top-down access.\n"
        "If blocking is unclear, prefer free.\n"
        "Return exactly one word: free or blocked."
    )


def build_target_id_user_text(task_text, labeled_text):
    visible_ids, id_lines = build_candidate_id_text(labeled_text)
    return (
        "Task: identify the target object in the numbered image.\n"
        f"Target instruction: {task_text}\n"
        f"Visible candidate IDs: {', '.join(visible_ids)}\n"
        "Numbered objects:\n"
        f"{id_lines}\n"
        "Match the described category and color.\n"
        "Do not choose an object only because it is nearby.\n"
        "Return exactly one object ID and nothing else."
    )


def build_object_id_user_text(object_description, labeled_text, candidate_ids=None):
    visible_ids, id_lines = build_candidate_id_text(labeled_text)
    candidate_line = ""
    if candidate_ids:
        candidate_line = f"Allowed candidate IDs: {', '.join(str(i) for i in candidate_ids)}\n"
    return (
        "Task: identify the described object in the numbered image.\n"
        f"Object description: {object_description}\n"
        f"Visible candidate IDs: {', '.join(visible_ids)}\n"
        f"{candidate_line}"
        "Numbered objects:\n"
        f"{id_lines}\n"
        "Choose only from the allowed candidate IDs if they are provided.\n"
        "Return exactly one matching object ID and nothing else."
    )


def build_original_scene_user_text(task_text):
    return (
        "Task: analyze the original scene image.\n"
        f"Target instruction: {task_text}\n"
        "Return 'free' if the target is directly graspable.\n"
        "Otherwise return 'blocked: <short blocker description>'.\n"
        "Use a short blocker description with color and category when possible."
    )


def build_blocker_id_user_text(task_text, labeled_text, blocker_ids=None):
    visible_ids, id_lines = build_candidate_id_text(labeled_text)
    blocker_line = ""
    if blocker_ids:
        blocker_line = f"Blocking candidate IDs: {', '.join(str(i) for i in blocker_ids)}\n"
    return (
        "Task: choose one blocker object.\n"
        "The target object is blocked and cannot be grasped directly.\n"
        f"Target instruction: {task_text}\n"
        f"Visible candidate IDs: {', '.join(visible_ids)}\n"
        f"{blocker_line}"
        "Numbered objects:\n"
        f"{id_lines}\n"
        "Choose only from the blocking candidate IDs if they are provided.\n"
        "A blocker sits on the target, crosses over it, covers its main graspable area, or prevents top-down access.\n"
        "Do not return the target object's own ID.\n"
        "Return exactly one blocker ID and nothing else."
    )


def parse_labeled_points(labeled_text):
    if isinstance(labeled_text, list):
        lines = labeled_text[1:]
    else:
        lines = labeled_text.strip().splitlines()[1:]
    points = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit():
            points.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return points


def infer_task_class_name(text):
    cleaned = text.strip().lower()
    cleaned = re.sub(r"^(take|pick|grasp|grab|remove)\s+", "", cleaned)
    cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned)
    return cleaned.strip()


def apply_state_memory_context(task_text, user_text):
    target_name = infer_task_class_name(task_text)
    memory_context = state_memory.build_memory_context(target_name, task_text)
    if memory_context:
        return f"{memory_context}\n\n{user_text}"
    return user_text


def update_state_memory(scene_objects=None, target_name=None, occlusion=None, grasp_success=None, region_hint=None, reason=None, scene_id=None):
    if scene_objects is not None or target_name is not None or occlusion is not None:
        state_memory.update_scene(scene_objects, target_name, occlusion)
    if grasp_success is not None and target_name:
        state_memory.record_grasp_result(
            target_name=target_name,
            region_hint=region_hint,
            success=grasp_success,
            reason=reason or "reasoning update",
            scene_id=scene_id,
        )
    return state_memory


def map_output_to_id(output, labeled_text):
    points = parse_labeled_points(labeled_text)
    if not points:
        return None

    coord_match = re.search(r"\[\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\]", output)
    if coord_match is None:
        coord_match = re.search(r"^\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s*$", output)
    if coord_match is None:
        return None

    x = float(coord_match.group(1))
    y = float(coord_match.group(2))
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    if x <= 1.0 and y <= 1.0:
        x *= max(xs)
        y *= max(ys)

    best_id = None
    best_dist = float("inf")
    for point_id, px, py in points:
        dist = (px - x) ** 2 + (py - y) ** 2
        if dist < best_dist:
            best_dist = dist
            best_id = point_id
    return best_id

def get_coordinates(labeled_text, goal_id):
    """
    Extract the (X, Y) coordinates for a given goal_id.
    """
    if isinstance(labeled_text, str):
        lines = labeled_text.strip().split("\n")[1:]  # Skip the header
    elif isinstance(labeled_text, list):
        lines = labeled_text[1:]
    
    for line in lines:
        parts = line.split()
        if int(parts[0]) == goal_id:
            return int(parts[1]), int(parts[2])
    return None  # Return None if not found


def decode_base64_image(base64_image):
    return Image.open(io.BytesIO(base64.b64decode(base64_image))).convert("RGB")


def encode_pil_image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def crop_image_around_point(base64_image, point, crop_size=320):
    image = decode_base64_image(base64_image)
    width, height = image.size
    x, y = point
    half = crop_size // 2
    left = max(0, x - half)
    top = max(0, y - half)
    right = min(width, x + half)
    bottom = min(height, y + half)

    if right - left < crop_size:
        if left == 0:
            right = min(width, crop_size)
        elif right == width:
            left = max(0, width - crop_size)
    if bottom - top < crop_size:
        if top == 0:
            bottom = min(height, crop_size)
        elif bottom == height:
            top = max(0, height - crop_size)

    cropped = image.crop((left, top, right, bottom))
    return encode_pil_image_to_base64(cropped), (left, top, right, bottom)


def build_local_labeled_text(labeled_text, crop_box, margin=48):
    left, top, right, bottom = crop_box
    lines = ["Molmo_ID X Y"]
    for point_id, px, py in parse_labeled_points(labeled_text):
        if left - margin <= px <= right + margin and top - margin <= py <= bottom + margin:
            lines.append(f"{point_id} {px} {py}")
    if len(lines) == 1:
        return labeled_text
    return "\n".join(lines)


def get_visible_ids(labeled_text):
    return [point_id for point_id, _, _ in parse_labeled_points(labeled_text)]


def get_local_candidate_distances(local_labeled_text, target_id, global_labeled_text=None):
    local_points = parse_labeled_points(local_labeled_text)
    if not local_points:
        return []

    target_point = get_coordinates(global_labeled_text or local_labeled_text, target_id)
    if target_point is None:
        return []

    tx, ty = target_point
    distances = []
    for point_id, px, py in local_points:
        if point_id == target_id:
            continue
        dist = (px - tx) ** 2 + (py - ty) ** 2
        distances.append((point_id, dist))
    distances.sort(key=lambda item: item[1])
    return distances


def get_blocker_distance_threshold(local_labeled_text):
    points = parse_labeled_points(local_labeled_text)
    if len(points) < 2:
        return float("inf")

    xs = [px for _, px, _ in points]
    ys = [py for _, _, py in points]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    span = max(span, 80)
    return (0.35 * span) ** 2


def get_blocking_candidates(local_labeled_text, target_id, global_labeled_text=None):
    local_points = parse_labeled_points(local_labeled_text)
    target_point = get_coordinates(global_labeled_text or local_labeled_text, target_id)
    if not local_points or target_point is None:
        return []

    tx, ty = target_point
    xs = [px for _, px, _ in local_points]
    ys = [py for _, _, py in local_points]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 80)
    horiz_thresh = 0.4 * span
    vert_thresh = 0.45 * span

    candidates = []
    for point_id, px, py in local_points:
        if point_id == target_id:
            continue
        dx = abs(px - tx)
        dy = ty - py
        # Only treat objects above the target, or nearly level but slightly above, as blockers.
        if dy < -0.05 * span:
            continue
        if dx > horiz_thresh:
            continue
        if dy > vert_thresh:
            continue
        score = (dx / max(horiz_thresh, 1.0)) + (max(dy, 0.0) / max(vert_thresh, 1.0))
        candidates.append((point_id, score))

    candidates.sort(key=lambda item: item[1])
    return [point_id for point_id, _ in candidates]


def get_goal_mask_with_index(masks, goal_coor):
    """
    Select the best mask (and its index) that contains the goal coordinate.
    If none contain the coordinate, select the closest one.
    """
    x, y = goal_coor
    best_mask = None
    selected_index = None
    min_distance = float('inf')

    for idx, mask in enumerate(masks):
        mask_np = mask.cpu().numpy()  # Convert to NumPy array
        if mask_np[y, x]:
            return mask, idx

        # If not inside, compute the minimum distance from goal_coor to mask pixels
        mask_indices = np.column_stack(np.where(mask_np > 0))
        if mask_indices.size:
            distances = np.linalg.norm(mask_indices - np.array([y, x]), axis=1)
            min_dist = np.min(distances)
            if min_dist < min_distance:
                min_distance = min_dist
                best_mask = mask
                selected_index = idx

    return best_mask, selected_index


def visualize_cropping_box(image, cropping_box):
    # Visualize the cropping box on the image
    x1, y1, x2, y2 = cropping_box
    plt.figure()
    plt.imshow(image)
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1,
                                      y2-y1, edgecolor='red', facecolor='none'))
    plt.title("Cropping Box Visualization")
    plt.show()


# Helper functions
def load_image_as_base64(image_path):
    with open(image_path, 'rb') as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_image



def process_grasping_result(output, text, labeled_text=None):
    """
    Parses the grasping result output and extracts object ID and class name.

    Supports two formats:
    - "[ID, class name]"  -> Example: "[1, green cylinder]"
    - "[pick object, ID, class name]"  -> Example: "[pick object, 4, blue bolt]"
    """

    match = re.search(r"\b(\d+)\b", output)

    if match:
        object_id = int(match.group(1))
        class_name = infer_task_class_name(text)
    else:
        mapped_id = map_output_to_id(output, labeled_text) if labeled_text is not None else None
        if mapped_id is not None:
            return {
                "selected_object_id": mapped_id,
                "class_name": infer_task_class_name(text)
            }
        class_name = infer_task_class_name(text)
        return {"class_name": class_name}
    
    return {
        "selected_object_id": object_id,  # Target object ID
        "class_name": class_name  # Target object class name
    }


def _run_reasoning_request(system_prompt, user_text, base64_image):
    return reasoning_client.generate(
        [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                ],
            },
        ],
        temperature=0.0,
        max_new_tokens=64,
        top_p=1.0,
        num_beams=1,
        use_cache=True,
    )


def classify_target_reachability(input_text, base64_image, labeled_text=None):
    user_text = build_reachability_user_text(input_text, labeled_text or "")
    user_text = apply_state_memory_context(input_text, user_text)
    output = _run_reasoning_request(REACHABILITY_SYSTEM_PROMPT, user_text, base64_image)
    lowered = output.strip().lower()
    if "blocked" in lowered:
        return "blocked", output
    if "free" in lowered:
        return "free", output
    return "free", output


def analyze_original_scene(input_text, original_base64_image):
    user_text = build_original_scene_user_text(input_text)
    user_text = apply_state_memory_context(input_text, user_text)
    output = _run_reasoning_request(
        ORIGINAL_SCENE_SYSTEM_PROMPT,
        user_text,
        original_base64_image,
    ).strip()
    lowered = output.lower()
    if lowered.startswith("free"):
        return "free", None, output
    if lowered.startswith("blocked:"):
        blocker_desc = output.split(":", 1)[1].strip()
        return "blocked", blocker_desc if blocker_desc else None, output
    if "blocked" in lowered:
        blocker_desc = output.split(":", 1)[1].strip() if ":" in output else None
        return "blocked", blocker_desc if blocker_desc else None, output
    return "free", None, output


def identify_object_id_from_description(description, labeled_base64_image, labeled_text, candidate_ids=None):
    user_text = build_object_id_user_text(description, labeled_text, candidate_ids)
    user_text = apply_state_memory_context(description, user_text)
    output = _run_reasoning_request(
        TARGET_ID_SYSTEM_PROMPT,
        user_text,
        labeled_base64_image,
    )
    match = re.search(r"\b(\d+)\b", output)
    if match:
        object_id = int(match.group(1))
        if candidate_ids and object_id not in set(candidate_ids):
            return None, output
        return object_id, output
    return None, output


def run_reasoning_model(input_text, labeled_base64_image, labeled_text=None, original_base64_image=None):
    target_id, _ = identify_object_id_from_description(
        input_text,
        labeled_base64_image,
        labeled_text or "",
    )

    focus_image_b64 = labeled_base64_image
    focus_labeled_text = labeled_text or ""
    if target_id is not None:
        target_point = get_coordinates(labeled_text or "", target_id)
        if target_point is not None:
            focus_image_b64, crop_box = crop_image_around_point(labeled_base64_image, target_point)
            focus_labeled_text = build_local_labeled_text(labeled_text or "", crop_box)

    if original_base64_image is None:
        original_base64_image = labeled_base64_image

    scene_state, blocker_desc, _ = analyze_original_scene(input_text, original_base64_image)

    blocker_ids = []
    if scene_state == "blocked" and target_id is not None:
        blocker_ids = get_blocking_candidates(
            focus_labeled_text,
            target_id,
            labeled_text or focus_labeled_text,
        )

    if not blocker_ids:
        if target_id is not None:
            return str(target_id)
        user_text = build_target_id_user_text(input_text, focus_labeled_text)
        user_text = apply_state_memory_context(input_text, user_text)
        output = _run_reasoning_request(TARGET_ID_SYSTEM_PROMPT, user_text, focus_image_b64)
    else:
        blocker_id = None
        if blocker_desc:
            blocker_id, _ = identify_object_id_from_description(
                blocker_desc,
                focus_image_b64,
                focus_labeled_text,
                blocker_ids,
            )
        if blocker_id is not None:
            return str(blocker_id)
        return str(blocker_ids[0])

    match = re.search(r"\b(\d+)\b", output)
    if match:
        return str(int(match.group(1)))
    return output



def create_cropping_box_from_boxes(box, image_size, margin=20):
    if len(box) == 0:
        return 0, 0, image_size[0], image_size[1]

    x1_min = float('inf')
    y1_min = float('inf')
    x2_max = float('-inf')
    y2_max = float('-inf')

    x1, y1, x2, y2 = box
    if x1 < x1_min:
        x1_min = x1
    if y1 < y1_min:
        y1_min = y1
    if x2 > x2_max:
        x2_max = x2
    if y2 > y2_max:
        y2_max = y2

    x1_min = max(0, x1_min - margin)
    y1_min = max(0, y1_min - margin)
    x2_max = min(image_size[0], x2_max + margin)
    y2_max = min(image_size[1], y2_max + margin)

    return int(x1_min), int(y1_min), int(x2_max), int(y2_max)



def get_and_process_data(cropping_box, color, depth, camera, viz):
    kernel = 0.2

    color = Image.fromarray(color)
    color = np.array(color, dtype=np.float32) / 255.0

    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    x1, y1, x2, y2 = map(int, cropping_box)
    x1_, y1_, x2_, y2_ = x1 - int((x2 - x1) * kernel) - 50, y1 - int((y2 - y1) * kernel) - 50, x2 + int(
        (x2 - x1) * kernel) + 50, y2 + int((y2 - y1) * kernel) + 50

    xmin, ymin, xmax, ymax = 0, 0, camera.width, camera.height
    dx1, dy1, dx2, dy2 = max(x1_, xmin), max(
        y1_, ymin), min(x2_, xmax), min(y2_, ymax)
    print(x1_, y1_, x2_, y2_, xmin, ymin, xmax, ymax)

    mask = np.zeros_like(depth)

    mask[dy1:dy2, dx1:dx2] = 1
    mask = (mask > 0) & (depth > 0)
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    print("Point cloud number", len(cloud_masked))

    num_point = 20000
    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(
            len(cloud_masked), num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_sampled.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_sampled.astype(np.float32))

    end_points = dict()
    cloud_sampled = torch.from_numpy(
        cloud_sampled[np.newaxis].astype(np.float32))
    cloud_sampled = cloud_sampled.to(torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'))  

    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled
    if viz:
        o3d.visualization.draw_geometries([cloud])

    return end_points, cloud



### REASONING UTILS ### 
def parse_points_with_id(file_path):
    """Reads the saved sceneXXX_id.txt file and extracts Molmo IDs with GT IDs directly."""
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: {file_path} not found!")

    with open(file_path, 'r') as file:
        lines = file.readlines()

    parsed_data = []
    molmo_to_gt_map = {}  # Direct mapping from Molmo ID → GT ID

    for line in lines[1:]:  # Skip header
        parts = line.split()
        if len(parts) == 4:  # Ensure correct format: "Molmo_ID X Y GT_ID"
            molmo_id, x, y, gt_id = map(int, parts)
            parsed_data.append((molmo_id, x, y, gt_id))
            molmo_to_gt_map[molmo_id] = gt_id  # Store direct mapping

    return parsed_data, molmo_to_gt_map, lines


def get_prediction_result(image, path, scene_number, text):
    try:
        scene_path = os.path.join(path)
        image_path = os.path.join(scene_path, f"{scene_number}.png")
        id_file_path = os.path.join(scene_path, f"{scene_number}_id.txt")

        image_pil = image 
        # image_pil = langsamutils.load_image(str(ori_image_path))
        
        # Load labeled image as base64
        base64_labeled_image = load_image_as_base64(image_path)
        base64_original_image = encode_pil_image_to_base64(image_pil)
        
        # Read object ID file
        parsed_data, molmo_to_gt_map, points_with_ids = parse_points_with_id(id_file_path)

        # GPT input text
        input_text = f"Grasp {text}"
        target_name = infer_task_class_name(text)
        update_state_memory(target_name=target_name, occlusion=None)

        output = run_reasoning_model(input_text, base64_labeled_image, points_with_ids, base64_original_image)
        result = process_grasping_result(output, text, points_with_ids)
        

        if 'selected_object_id' not in result or result["selected_object_id"] not in [item[0] for item in parsed_data]:
            print("No object id detected by GPT, using LangSAM for segmentation...")
            input2LangSAM = result["class_name"]
            masks, boxes, phrases, logits = langsam_actor.predict(image_pil, input2LangSAM)
            
            if masks is None or masks.numel() == 0:
                raise Exception("LangSAM failed to segment the object.")

            best_index = np.argmax(logits.cpu().numpy())
            goal_mask = masks[best_index]
            goal_bbox = boxes[best_index].cpu().numpy()
        else:
            goal = result["class_name"]
            goal_id = result['selected_object_id']
            goal_coor = get_coordinates(points_with_ids, goal_id)
            masks, boxes, phrases, logits = langsam_actor.predict(image_pil, goal)

            # Selected the mask of the object we want based on the coordinates generate from molmo (if there are multi same class objects)
            goal_mask, mask_index = get_goal_mask_with_index(masks, goal_coor)
            if goal_mask is None or mask_index is None:
                raise Exception("LangSAM produced masks, but no valid goal mask matched the selected object ID.")
            goal_bbox = boxes[mask_index]
            goal_bbox = goal_bbox.cpu().numpy()

        goal_mask_np = goal_mask.cpu().numpy() if torch.is_tensor(goal_mask) else np.asarray(goal_mask)
        if goal_mask_np is None or goal_mask_np.ndim != 2:
            raise Exception(f"Invalid goal mask shape: {None if goal_mask_np is None else goal_mask_np.shape}")

        return goal_mask_np

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
