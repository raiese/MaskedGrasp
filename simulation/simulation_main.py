import argparse
import urllib.request
import urllib.error
import json
import logging
import threading
import re
import io
import pybullet as p
import numpy as np
import random

import ray
import torch
import os
import cv2
from matplotlib import pyplot as plt
import open3d as o3d
import wandb
from PIL import Image
from scipy.spatial.transform import Rotation as SciRot
import utils
from constants import WORKSPACE_LIMITS
from environment_sim import Environment
from langsam import langsamutils
from langsam.langsam_actor import LangSAM
from logger import Logger
from grasp_detetor import Graspnet
# from VLP.new_vlp_actor import SegmentAnythingActor

import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


QWEN_SYSTEM_PROMPT_BASE = (
    "You are the perception module for a robotic grasping simulator.\n"
    "You receive one 224x224 top-down RGB image and one natural-language instruction.\n"
    "Choose the single best current grasp target from the CURRENT image only.\n"
    "\n"
    "Rules:\n"
    "1. Focus only on the current image. Do not repeat previous answers unless the scene truly still looks the same.\n"
    "2. If the instructed target object is visible, select that object.\n"
    "3. If the target is partially occluded but still identifiable, still select it.\n"
    "4. If the target cannot be identified confidently, select the single best blocking object to move first.\n"
    "5. Prefer safer and easier graspable objects or object parts.\n"
    "6. If an object has an easier graspable part, you may select that part.\n"
    "7. Do not invent material, texture, stability, or other long descriptions.\n"
    "8. The cropping box must tightly cover the selected object or selected object part.\n"
    "9. Do not output (0, 0, 224, 224) unless the object truly occupies most of the image or cannot be localized more precisely.\n"
    "10. Coordinates must be integer pixel coordinates in the 224x224 image.\n"
    "11. Preferred Grasping Location must be a single number from 1 to 9 using the 3x3 grid inside the cropping box:\n"
    "1 2 3\n"
    "4 5 6\n"
    "7 8 9\n"
    "12. For round objects such as balls, prefer the center unless one side is clearly less occluded.\n"
    "13. If uncertain, make the most conservative reasonable choice from the visible objects.\n"
    "\n"
    "Output exactly these three lines and nothing else:\n"
    "Selected Object/Object Part: [object:color and name] or [object part:color and name]\n"
    "Cropping Box Coordinates: (x1, y1, x2, y2)\n"
    "Preferred Grasping Location: [1-9]"
)

QWEN_SYSTEM_PROMPT_STRICT = (
    QWEN_SYSTEM_PROMPT_BASE
    + "\n"
    + "Additional strict rules:\n"
    + "14. You must choose exactly ONE target. Never output multiple options. Never use the word 'or'.\n"
    + "15. The first line must contain exactly one bracketed item.\n"
    + "16. Use only these two labels:\n"
    + "[object:...]\n"
    + "[object part:...]\n"
    + "17. Do not use labels such as [ball:...], [cup:...], [target:...], or any other custom label.\n"
    + "18. Inside the brackets, output only a short natural phrase such as [object:yellow ball] or [object part:mug handle].\n"
    + "19. Do not output color lists like 'yellow and green' unless both colors are clearly visible on the same selected object.\n"
    + "20. If the instruction is generic, such as 'grasp a ball', choose the single most graspable visible ball.\n"
    + "21. Keep the cropping box tightly around that one selected target only.\n"
    + "22. Do not output explanations, alternatives, confidence scores, or any extra text.\n"
    + "23. The object name must include an object category noun, not just a color.\n"
    + "24. Bad examples:\n"
    + "[object:yellow]\n"
    + "[object:white]\n"
    + "[object:green]\n"
    + "[ball:yellow]\n"
    + "25. Good examples:\n"
    + "[object:yellow ball]\n"
    + "[object:white ball]\n"
    + "[object:red bottle]\n"
    + "[object:yellow cup]\n"
    + "26. If the object is a ball, the output must explicitly contain the word 'ball'.\n"
    + "27. If the instruction is 'grasp a ball', never output only a color. Always output a full object phrase such as [object:yellow ball].\n"
)


class VisionLanguageClient:
    def __init__(self, backend, model_name, model_path, device, api_url=None, system_prompt=None):
        self.backend = backend
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.api_url = api_url
        self.system_prompt = system_prompt or QWEN_SYSTEM_PROMPT_BASE
        self.client = None
        self.model = None
        self.processor = None
        self._vision_info_fn = None

        if backend == "openai":
            from openai import OpenAI

            api_key = os.environ["OPENAI_API_KEY"]
            self.client = OpenAI(api_key=api_key)
        elif backend == "qwen_local":
            from transformers import AutoProcessor

            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
            except ImportError:
                try:
                    from transformers import Qwen2VLForConditionalGeneration as QwenVLModel
                except ImportError:
                    from transformers import AutoModelForVision2Seq as QwenVLModel
            try:
                from qwen_vl_utils import process_vision_info
            except ImportError:
                process_vision_info = None

            model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.model = QwenVLModel.from_pretrained(
                model_path,
                torch_dtype=model_dtype,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(model_path)
            self._vision_info_fn = process_vision_info
        elif backend == "qwen_remote":
            if not api_url:
                raise ValueError("vlm_api_url is required when using qwen_remote backend")
        else:
            raise ValueError(f"Unsupported VLM backend: {backend}")

    def generate(self, instruction, image_path, max_new_tokens=768):
        if self.backend == "openai":
            base64_image = load_image_as_base64(image_path)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    ],
                },
            ]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=max_new_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return response.choices[0].message.content

        if self.backend == "qwen_remote":
            payload = json.dumps({
                "instruction": instruction,
                "image_path": "data:image/png;base64," + load_image_as_base64(image_path),
                "system_prompt": self.system_prompt,
                "max_new_tokens": max_new_tokens,
            }).encode("utf-8")
            request = urllib.request.Request(
                self.api_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    body = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Qwen remote request failed: HTTP {exc.code}: {error_body}") from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(
                    f"Failed to reach Qwen remote service at {self.api_url}: {exc}"
                ) from exc

            if "error" in body:
                raise RuntimeError(f"Qwen remote inference failed: {body['error']}")
            return body["text"]

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": instruction},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if self._vision_info_fn is not None:
            image_inputs, video_inputs = self._vision_info_fn(messages)
        else:
            image_inputs = [Image.open(image_path).convert("RGB")]
            video_inputs = None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output[0]


def select_action(bboxes, pos_bboxes, text, actions, evaluate=True):
    distances = torch.sqrt((actions[0, :, 0] - pos_bboxes[0, 0, 0]) ** 2 +
                           (actions[0, :, 1] - pos_bboxes[0, 0, 1]) ** 2)

    num_distances_to_select = min(10, distances.numel())
    top_dist_indices = torch.topk(distances, k=num_distances_to_select, largest=False).indices
    min_dist_index = top_dist_indices[0]

    return min_dist_index.item()

def load_image_as_base64(image_path):
    with open(image_path, 'rb') as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_image


def encode_rgb_image_to_base64(image_rgb):
    success, encoded = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("Failed to encode RGB image for FreeGrasp request.")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def encode_depth_npz_to_base64(depth_image):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, depth=depth_image.astype(np.float32))
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_camera_intrinsics_payload(camera_cfg):
    intr = camera_cfg["intrinsics"]
    height, width = camera_cfg["image_size"]
    return {
        "width": int(width),
        "height": int(height),
        "fx": float(intr[0, 0]),
        "fy": float(intr[1, 1]),
        "cx": float(intr[0, 2]),
        "cy": float(intr[1, 2]),
        "scale": 1.0,
    }


def build_camera_to_world(camera_cfg):
    rotation = np.array(p.getMatrixFromQuaternion(camera_cfg["rotation"]), dtype=np.float64).reshape(3, 3)
    translation = np.asarray(camera_cfg["position"], dtype=np.float64)
    return rotation, translation


def freegrasp_pose_to_world_action(camera_cfg, grasp_pose):
    grasp_rot_cam = np.asarray(grasp_pose["rotation"], dtype=np.float64)
    grasp_trans_cam = np.asarray(grasp_pose["translation"], dtype=np.float64)
    cam_rot, cam_trans = build_camera_to_world(camera_cfg)
    world_rot = cam_rot @ grasp_rot_cam
    world_trans = cam_rot @ grasp_trans_cam + cam_trans
    quat = SciRot.from_matrix(world_rot).as_quat()
    action = np.concatenate([world_trans, quat]).astype(np.float32)
    return action, world_rot, world_trans


def request_freegrasp_remote(api_url, color_image, depth_image, camera_cfg, task_text, request_id, return_artifacts=False):
    payload = {
        "request_id": request_id,
        "task": task_text,
        "camera_intrinsics": get_camera_intrinsics_payload(camera_cfg),
        "files": {
            "image_png_base64": encode_rgb_image_to_base64(color_image),
            "depth_npz_base64": encode_depth_npz_to_base64(depth_image),
        },
        "return_artifacts": bool(return_artifacts),
    }
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"FreeGrasp remote request failed: HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach FreeGrasp remote service at {api_url}: {exc}") from exc

    if not body.get("ok", False):
        raise RuntimeError(body.get("error", "FreeGrasp remote inference failed."))
    if "grasp_pose" not in body or not body["grasp_pose"]:
        raise RuntimeError("FreeGrasp remote response did not include a grasp pose.")
    return body

def print_grasping_result(result):
    logging.info(f"Selected Object/Object Part: {result['selected_object']}")
    logging.info(f"Cropping Box Coordinates: {result['cropping_box']}")
    logging.info("Objects and Their Properties:")
    for obj in result["objects"]:
        logging.info(f"Object: {obj['name']}")
        logging.info(f"  Grasping Score: {obj['grasping_score']}")
        logging.info(f"  Material Composition: {obj['material_composition']}")
        logging.info(f"  Surface Texture: {obj['surface_texture']}")
        logging.info(f"  Stability Assessment: {obj['stability_assessment']}")
        logging.info(f"  Centroid Coordinates: {obj['centroid_coordinates']}")
        logging.info(f"  Preferred Grasping Location: {obj['preferred_grasping_location']}")
        logging.info("")


def append_diagnostic_log(log_path, payload):
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(payload, ensure_ascii=True) + "\n")


def create_cropping_box_from_boxes(boxes, image_size, margin=20):
    if not boxes:
        # If boxes list is empty, return a default cropping box (e.g., full image)
        return 0, 0, image_size[0], image_size[1]

    # Create a cropping box that includes all detected bounding boxes
    x1_min = float('inf')
    y1_min = float('inf')
    x2_max = float('-inf')
    y2_max = float('-inf')

    for box in boxes:
        x1, y1, x2, y2 = box
        if x1 < x1_min:
            x1_min = x1
        if y1 < y1_min:
            y1_min = y1
        if x2 > x2_max:
            x2_max = x2
        if y2 > y2_max:
            y2_max = y2

    # Add margin to the bounding box
    x1_min = max(0, x1_min - margin)
    y1_min = max(0, y1_min - margin)
    x2_max = min(image_size[0], x2_max + margin)
    y2_max = min(image_size[1], y2_max + margin)

    return int(x1_min), int(y1_min), int(x2_max), int(y2_max)


def crop_pointcloud(pcd, cropping_box, color_image, depth_image, workspace_limits):
    # Convert the 2D cropping box coordinates to 3D coordinates
    x1, y1, x2, y2 = cropping_box
    depth_crop = depth_image[y1:y2, x1:x2]
    color_crop = color_image[y1:y2, x1:x2]
    mask = (depth_crop > 0)
    points = []
    colors = []

    for i in range(depth_crop.shape[0]):
        for j in range(depth_crop.shape[1]):
            if mask[i, j]:
                x = j + x1
                y = i + y1
                z = depth_crop[i, j]
                points.append((x, y, z))
                colors.append(color_crop[i, j] / 255.0)  # Normalize the color values

    points = np.array(points)
    colors = np.array(colors)

    # Convert image coordinates to workspace coordinates
    def image_to_workspace(x, y, z, image_width, image_height, workspace_limits):
        workspace_x = workspace_limits[0][0] + (x / image_width) * (workspace_limits[0][1] - workspace_limits[0][0])
        workspace_y = workspace_limits[1][0] + (y / image_height) * (workspace_limits[1][1] - workspace_limits[1][0])
        workspace_z = z  # Assuming z is already in the correct unit (e.g., meters)
        return workspace_x, workspace_y, workspace_z

    image_height, image_width = depth_image.shape
    workspace_points = [image_to_workspace(x, y, z, image_width, image_height, workspace_limits) for x, y, z in points]

    workspace_points = np.array(workspace_points)

    # Create a grid of points for the entire workspace with z=0
    grid_x, grid_y = np.meshgrid(
        np.linspace(workspace_limits[0][0], workspace_limits[0][1], image_width),
        np.linspace(workspace_limits[1][0], workspace_limits[1][1], image_height)
    )
    grid_z = np.zeros_like(grid_x)
    full_workspace_points = np.stack((grid_x, grid_y, grid_z), axis=-1).reshape(-1, 3)
    full_colors = np.zeros((full_workspace_points.shape[0], 3))

    # Update the points in the cropped region with actual height
    for idx, (x, y, z) in enumerate(workspace_points):
        # Convert x, y from workspace coordinates back to image coordinates to find the corresponding index
        image_x = int((x - workspace_limits[0][0]) / (workspace_limits[0][1] - workspace_limits[0][0]) * image_width)
        image_y = int((y - workspace_limits[1][0]) / (workspace_limits[1][1] - workspace_limits[1][0]) * image_height)

        # Ensure the indices are within bounds
        if 0 <= image_x < image_width and 0 <= image_y < image_height:
            grid_index = image_x * image_height + image_y  # Correct index calculation
            full_workspace_points[grid_index, 2] = z
            full_colors[grid_index] = colors[idx]

    # Create a new point cloud from the full workspace points
    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(full_workspace_points)
    full_pcd.colors = o3d.utility.Vector3dVector(full_colors)

    return full_pcd

def visualize_cropping_box(image, cropping_box):
    # Visualize the cropping box on the image
    x1, y1, x2, y2 = cropping_box
    plt.figure()
    plt.imshow(image)
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', facecolor='none'))
    plt.title("Cropping Box Visualization")
    plt.show()



def visualize_pointcloud_with_grasps(pcd, grasp_poses):
    # Visualize the point cloud with grasp poses
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    for grasp in grasp_poses:
        grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        grasp_frame.translate(grasp.translation)
        grasp_frame.rotate(grasp.rotation_matrix)
        vis.add_geometry(grasp_frame)

    vis.run()
    vis.destroy_window()
def process_grasping_result(output):
    lines = output.strip().split('\n')
    result = {
        "selected_object": None,
        "cropping_box": None,
        "objects": [],
        "is_part": False  # New field to indicate if the selected object is a part
    }

    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if line.startswith("Selected Object/Object Part:") or (
            i == 0 and stripped_line.startswith("[") and stripped_line.endswith("]")
        ):
            if line.startswith("Selected Object/Object Part:"):
                selected_object = line.split(": ", 1)[1].strip()
            else:
                selected_object = stripped_line
            bracket_match = re.match(r"^\[(.+?)\]$", selected_object)
            if bracket_match:
                inner = bracket_match.group(1)
                if ":" in inner:
                    label, value = inner.split(":", 1)
                    result["is_part"] = "part" in label.lower()
                    selected_object = value.strip()
                else:
                    result["is_part"] = "part" in inner.lower()
                    selected_object = inner.strip()
            else:
                result["is_part"] = "part" in selected_object.lower()
            result["selected_object"] = selected_object


        elif line.startswith("Cropping Box Coordinates:"):
            coords = line.split(": ")[1].strip()[1:-1]
            result["cropping_box"] = tuple(map(int, coords.split(", ")))
        elif line.startswith("Preferred Grasping Location:"):
            try:
                preferred_location = int(line.split(": ")[1].strip())
            except ValueError:
                preferred_location = 5
            if result["selected_object"] and not result["objects"]:
                result["objects"].append({
                    "name": result["selected_object"],
                    "grasping_score": None,
                    "material_composition": None,
                    "surface_texture": None,
                    "stability_assessment": None,
                    "centroid_coordinates": None,
                    "preferred_grasping_location": preferred_location,
                })
        elif line.startswith("Object:"):
            obj = {
                "name": line.split(": ")[1].strip(),
                "grasping_score": None,
                "material_composition": None,
                "surface_texture": None,
                "stability_assessment": None,
                "centroid_coordinates": None,
                "preferred_grasping_location": 5,
            }
            if i + 6 < len(lines):
                try:
                    obj["grasping_score"] = int(lines[i + 1].split(": ")[1].strip())
                    obj["material_composition"] = int(lines[i + 2].split(": ")[1].strip())
                    obj["surface_texture"] = int(lines[i + 3].split(": ")[1].strip())
                    obj["stability_assessment"] = int(lines[i + 4].split(": ")[1].strip())
                    obj["centroid_coordinates"] = tuple(map(int, lines[i + 5].split(": ")[1].strip()[1:-1].split(", ")))
                    obj["preferred_grasping_location"] = int(lines[i + 6].split(": ")[1].strip())
                except (IndexError, ValueError):
                    pass
            result["objects"].append(obj)

    return result



def select_fallback_object(objects):
    # Select the most cost-effective object to move
    sorted_objects = sorted(objects, key=lambda x: x['grasping_score'], reverse=True)
    return sorted_objects[0] if sorted_objects else None
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', action='store', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1234, metavar='N',
                    help='random seed (default: 1234)')
    parser.add_argument('--evaluate', action='store', type=bool, default=True)
    parser.add_argument('--testing_case_dir', action='store', type=str, default='testing_cases/')
    parser.add_argument('--testing_case', action='store', type=str, default=None)

    parser.add_argument('--load_model', action='store', type=bool, default=False)
    parser.add_argument('--model_path', action='store', type=str, default='')
    parser.add_argument('--vlm_backend', type=str, default='qwen_remote',
                        choices=['qwen_local', 'qwen_remote', 'openai'])
    parser.add_argument('--vlm_model_name', type=str, default='gpt-4o-2024-05-13')
    parser.add_argument('--vlm_model_path', type=str,
                        default='../weights/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--vlm_max_new_tokens', type=int, default=768)
    parser.add_argument('--vlm_api_url', type=str, default='http://127.0.0.1:8008/generate')
    parser.add_argument('--vlm_prompt_variant', type=str, default='base',
                        choices=['base', 'strict'])
    parser.add_argument('--sam_type', type=str, default='vit_h')
    parser.add_argument('--sam_ckpt_path', type=str,
                        default='../weights/groundingdino/sam_vit_h_4b8939.pth')
    parser.add_argument('--groundingdino_config_path', type=str,
                        default='langsam/GroundingDINO_SwinB_local_bert.cfg.py')
    parser.add_argument('--groundingdino_ckpt_path', type=str,
                        default='../weights/groundingdino/groundingdino_swinb_cogcoor.pth')

    parser.add_argument('--num_episode', action='store', type=int, default=15)
    parser.add_argument('--grasp_backend', type=str, default='thinkgrasp',
                        choices=['thinkgrasp', 'freegrasp_remote'])
    parser.add_argument('--freegrasp_api_url', type=str, default='http://127.0.0.1:8000/infer_grasp')
    parser.add_argument('--freegrasp_camera_index', type=int, default=0)
    parser.add_argument('--freegrasp_return_artifacts', action='store_true', default=False)
    parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy every 10 episode (default: True)')
    parser.add_argument('--max_episode_step', type=int, default=50)
    parser.add_argument('--gui', action='store_true', default=False,
                        help='Enable GUI visualization (PyBullet viewer and matplotlib plots)')

    # Transformer paras
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--layers', type=int, default=1) # cross attention layer
    parser.add_argument('--heads', type=int, default=8)

    # SAC parameters
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    wandb.init(project="maskedgrasp-local", mode=os.environ.get("WANDB_MODE", "disabled"))
    system_prompt = (
        QWEN_SYSTEM_PROMPT_STRICT
        if args.vlm_prompt_variant == "strict"
        else QWEN_SYSTEM_PROMPT_BASE
    )
    langsam_actor = None
    vlm_client = None
    graspnet = None
    if args.grasp_backend == "thinkgrasp":
        ray.init(num_gpus=1)
        use_gpu = torch.cuda.is_available()
        gpu_allocation = 1 if use_gpu else 0
        actor_options = {"num_gpus": gpu_allocation}
        langsam_actor = LangSAM.options(**actor_options).remote(
            use_gpu=use_gpu,
            sam_type=args.sam_type,
            ckpt_path=args.sam_ckpt_path,
            groundingdino_config_path=args.groundingdino_config_path,
            groundingdino_ckpt_path=args.groundingdino_ckpt_path,
        )



    # set device and seed
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if args.grasp_backend == "thinkgrasp":
        vlm_client = VisionLanguageClient(
            backend=args.vlm_backend,
            model_name=args.vlm_model_name,
            model_path=args.vlm_model_path,
            device=args.device,
            api_url=args.vlm_api_url,
            system_prompt=system_prompt,
        )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # parameters
    num_episode = args.num_episode

    # load environment
    env = Environment(gui=args.gui)
    env.seed(args.seed)
    # load logger
    logger = Logger(case_dir=args.testing_case_dir)
    diagnostic_log_path = os.path.join(logger.base_directory, "step_diagnostics.jsonl")
    if args.grasp_backend == "thinkgrasp":
        graspnet = Graspnet()

    if os.path.exists(args.testing_case_dir):
        filelist = os.listdir(args.testing_case_dir)
        filelist.sort(key=lambda x:int(x[4:6]))
    if args.testing_case != None:
        filelist = [args.testing_case]
    case = 0
    iteration = 0
    for f in filelist:
        f = os.path.join(args.testing_case_dir, f)

        logger.episode_reward_logs = []
        logger.episode_step_logs = []
        logger.episode_success_logs = []
        for episode in range(num_episode):
            episode_reward = 0
            episode_steps = 0
            done = False
            reset = False

            reset_attempts = 0
            max_reset_attempts = 3
            while not reset:
                env.reset()
                reset, lang_goal = env.add_object_push_from_file(f)
                reset_attempts += 1
                print(
                    f"\033[032m Reset environment of episode {episode}, language goal {lang_goal}, "
                    f"attempt {reset_attempts}/{max_reset_attempts}, success={reset}\033[0m"
                )
                if not reset and reset_attempts >= max_reset_attempts:
                    print(
                        "\033[033m Scene did not fully settle during reset. "
                        "Proceeding with the latest loaded scene.\033[0m"
                    )
                    reset = True

            while not done:
                # check if one of the target objects is in the workspace:
                out_of_workspace = []
                for obj_id in env.target_obj_ids:
                    pos, _, _ = env.obj_info(obj_id)
                    if pos[0] < WORKSPACE_LIMITS[0][0] or pos[0] > WORKSPACE_LIMITS[0][1] \
                        or pos[1] < WORKSPACE_LIMITS[1][0] or pos[1] > WORKSPACE_LIMITS[1][1]:
                        out_of_workspace.append(obj_id)
                if len(out_of_workspace) == len(env.target_obj_ids):
                    print("\033[031m Target objects are not in the scene!\033[0m")
                    break

                if args.grasp_backend == "freegrasp_remote":
                    if args.freegrasp_camera_index < 0 or args.freegrasp_camera_index >= len(env.agent_cams):
                        raise IndexError(
                            f"freegrasp_camera_index={args.freegrasp_camera_index} out of range for {len(env.agent_cams)} cameras"
                        )

                    camera_cfg = env.agent_cams[args.freegrasp_camera_index]
                    color_image, depth_image, mask_image = env.render_camera(camera_cfg)
                    request_id = f"ep{episode:03d}_step{episode_steps + 1:03d}_iter{iteration:06d}"

                    try:
                        freegrasp_response = request_freegrasp_remote(
                            api_url=args.freegrasp_api_url,
                            color_image=color_image,
                            depth_image=depth_image,
                            camera_cfg=camera_cfg,
                            task_text=lang_goal,
                            request_id=request_id,
                            return_artifacts=args.freegrasp_return_artifacts,
                        )
                        grasp_pose = freegrasp_response["grasp_pose"]
                        action, world_rot, world_trans = freegrasp_pose_to_world_action(camera_cfg, grasp_pose)
                    except Exception as e:
                        logging.error(f"Error with FreeGrasp request: {e}")
                        reward = -1
                        done = False
                        step_result = {
                            "pose": None,
                            "grasp_success": False,
                            "grasped_obj_id": None,
                            "target_obj_ids": list(map(int, env.target_obj_ids)),
                            "target_hit": False,
                            "reward": reward,
                            "done": done,
                            "error": str(e),
                        }
                        env.last_step_info = step_result
                        append_diagnostic_log(
                            diagnostic_log_path,
                            {
                                "episode": episode,
                                "step": episode_steps + 1,
                                "lang_goal": lang_goal,
                                "grasp_backend": args.grasp_backend,
                                "camera_index": args.freegrasp_camera_index,
                                "step_result": step_result,
                            },
                        )
                        logging.info(
                            "Step outcome: backend=%s grasp_success=%s target_hit=%s grasped_obj_id=%s reward=%s done=%s",
                            args.grasp_backend,
                            False,
                            False,
                            None,
                            reward,
                            done,
                        )
                        iteration += 1
                        episode_steps += 1
                        episode_reward += reward
                        print("\033[034m Episode: {}, step: {}, reward: {}\033[0m".format(episode, episode_steps, round(reward, 2)))
                        wandb.log({"episode": episode, "step": episode_steps, "reward": reward})
                        if episode_steps == args.max_episode_step:
                            break
                        continue

                    reward, done = env.step(action)
                    step_diag = {
                        "episode": episode,
                        "step": episode_steps + 1,
                        "lang_goal": lang_goal,
                        "grasp_backend": args.grasp_backend,
                        "camera_index": args.freegrasp_camera_index,
                        "camera_intrinsics": get_camera_intrinsics_payload(camera_cfg),
                        "freegrasp_request_id": freegrasp_response.get("request_id"),
                        "freegrasp_request_dir": freegrasp_response.get("request_dir"),
                        "freegrasp_log": freegrasp_response.get("log"),
                        "freegrasp_traceback": freegrasp_response.get("traceback"),
                        "freegrasp_grasp_pose": grasp_pose,
                        "world_frame_rotation": world_rot.tolist(),
                        "world_frame_translation": world_trans.tolist(),
                        "chosen_action": action.tolist(),
                        "step_result": env.last_step_info,
                    }
                    append_diagnostic_log(diagnostic_log_path, step_diag)
                    logging.info(
                        "Step outcome: backend=%s grasp_success=%s target_hit=%s grasped_obj_id=%s reward=%s done=%s",
                        args.grasp_backend,
                        env.last_step_info.get("grasp_success"),
                        env.last_step_info.get("target_hit"),
                        env.last_step_info.get("grasped_obj_id"),
                        reward,
                        done,
                    )
                    iteration += 1
                    episode_steps += 1
                    episode_reward += reward
                    print("\033[034m Episode: {}, step: {}, reward: {}\033[0m".format(episode, episode_steps, round(reward, 2)))
                    wandb.log({"episode": episode, "step": episode_steps, "reward": reward})

                    if episode_steps == args.max_episode_step:
                        break
                    continue


                color_image, depth_image, mask_image = utils.get_true_heightmap(env)





                image = "color_map.png"
                image_pil = langsamutils.load_image(image)

                image_path = 'color_map.png'

                input_text = (
                    f"Instruction: {lang_goal}\n"
                    "Return only the required three-line output for the current image."
                )

                result = {
                    "selected_object": None,
                    "cropping_box": None,
                    "objects": []
                }

                try:
                    output = vlm_client.generate(
                        instruction=input_text,
                        image_path=image_path,
                        max_new_tokens=args.vlm_max_new_tokens,
                    )
                    logging.info(output)
                    result = process_grasping_result(output)
                    print_grasping_result(result)
                    with open('grasping_result_log.json', 'w') as log_file:
                        json.dump(result, log_file, indent=4)
                    wandb.log({f"{args.vlm_backend}_output": output})
                except Exception as e:
                    logging.error(f"Error with VLM request: {e}")


                finally:

                    if not result['selected_object']:

                        fallback_object = select_fallback_object(result['objects'])

                        if fallback_object:

                            goal = fallback_object['name']

                        else:

                            goal = lang_goal

                    else:

                        goal = result['selected_object']

                    preferred_grasping_location = 5  # Default to center if not specified
                    for obj in result['objects']:
                        if obj['name'] == goal:
                            preferred_grasping_location = obj.get('preferred_grasping_location', 5)

                    masks, boxes, phrases, logits = ray.get(langsam_actor.predict.remote(image_pil, goal))
                    if masks is None or masks.numel() == 0:
                        masks, boxes, phrases, logits = ray.get(langsam_actor.predict.remote(image_pil, lang_goal))
                        if masks is None or masks.numel() == 0:
                            masks, boxes, phrases, logits = ray.get(langsam_actor.predict.remote(image_pil, "object"))

                    phrase_values = [str(phrase) for phrase in phrases]
                    logit_values = [float(logit) for logit in logits] if hasattr(logits, "__iter__") else []

                    boxes_list = boxes.cpu().numpy().tolist()  # Convert to list
                    cropping_box = create_cropping_box_from_boxes(boxes_list,
                                                                  (color_image.shape[1], color_image.shape[0]))

                    if args.gui:
                        visualize_cropping_box(color_image, cropping_box)
                    ray.get(langsam_actor.save.remote(masks, boxes, phrases, logits, image_pil, gui=args.gui))
                    bbox_images, bbox_positions = utils.convert_output(image_pil, boxes, phrases, logits, color_image, depth_image, mask_image, preferred_grasping_location)

                    # graspnet
                    pcd = utils.get_fuse_pointcloud(env)
                    cropped_pcd = crop_pointcloud(pcd, cropping_box,color_image, depth_image, WORKSPACE_LIMITS)

                    if cropped_pcd is None or len(np.asarray(cropped_pcd.points)) == 0:
                        print("\033[031m Cropped point cloud is empty!\033[0m")
                        continue
                    # Visualize the cropped point cloud
                    # o3d.visualization.draw_geometries([pcd, cropped_pcd], window_name="Cropped Point Cloud")

                    # o3d.visualization.draw_geometries([ cropped_pcd], window_name="Cropped Point Cloud")
                    #

                    # Perform grasp detection on the cropped point cloud
                    with torch.no_grad():
                        grasp_pose_set, _, _ = graspnet.grasp_detection(cropped_pcd, env.get_true_object_poses())
                    print("Number of grasping poses", len(grasp_pose_set))
                    logging.info(f"Number of grasping poses: {len(grasp_pose_set)}")

                    if len(grasp_pose_set) == 0:
                        with torch.no_grad():
                            grasp_pose_set, _, _ = graspnet.grasp_detection(pcd, env.get_true_object_poses())
                        print("Number of grasping poses", len(grasp_pose_set))
                        logging.info(f"Number of grasping poses: {len(grasp_pose_set)}")
                        if len(grasp_pose_set) == 0:
                            break
                    # preprocess
                    remain_bbox_images, bboxes, pos_bboxes, grasps = utils.preprocess(bbox_images, bbox_positions, grasp_pose_set, (args.patch_size, args.patch_size))
                    logger.save_bbox_images(iteration, remain_bbox_images)
                    logger.save_heightmaps(iteration, color_image, depth_image)
                    if bboxes == None:
                        break

                    if len(grasp_pose_set) == 1:
                        action_idx = 0
                    else:
                        with torch.no_grad():
                            action_idx = select_action(bboxes, pos_bboxes, lang_goal, grasps)
                    action = grasp_pose_set[action_idx]




                    reward, done = env.step(action)
                    step_diag = {
                        "episode": episode,
                        "step": episode_steps + 1,
                        "lang_goal": lang_goal,
                        "vlm_selected_object": result.get("selected_object"),
                        "vlm_cropping_box": result.get("cropping_box"),
                        "goal_used_for_langsam": goal,
                        "preferred_grasping_location": preferred_grasping_location,
                        "langsam_phrases": phrase_values,
                        "langsam_logits": logit_values,
                        "langsam_boxes": boxes_list,
                        "grasp_pose_count": len(grasp_pose_set),
                        "chosen_action": action.tolist() if hasattr(action, "tolist") else action,
                        "step_result": env.last_step_info,
                    }
                    append_diagnostic_log(diagnostic_log_path, step_diag)
                    logging.info(
                        "Step outcome: goal=%s grasp_success=%s target_hit=%s grasped_obj_id=%s reward=%s done=%s",
                        goal,
                        env.last_step_info.get("grasp_success"),
                        env.last_step_info.get("target_hit"),
                        env.last_step_info.get("grasped_obj_id"),
                        reward,
                        done,
                    )
                    iteration += 1
                    episode_steps += 1
                    episode_reward += reward
                    print("\033[034m Episode: {}, step: {}, reward: {}\033[0m".format(episode, episode_steps, round(reward, 2)))
                    wandb.log({"episode": episode, "step": episode_steps, "reward": reward})

                    if episode_steps == args.max_episode_step:
                        break


            logger.episode_reward_logs.append(episode_reward)
            logger.episode_step_logs.append(episode_steps)
            logger.episode_success_logs.append(done)
            logger.write_to_log('episode_reward', logger.episode_reward_logs)
            logger.write_to_log('episode_step', logger.episode_step_logs)
            logger.write_to_log('episode_success', logger.episode_success_logs)
            print("\033[034m Episode: {}, episode steps: {}, episode reward: {}, success: {}\033[0m".format(episode, episode_steps, round(episode_reward, 2), done))

            if episode == num_episode - 1:
                avg_success = sum(logger.episode_success_logs)/len(logger.episode_success_logs)
                avg_reward = sum(logger.episode_reward_logs)/len(logger.episode_reward_logs)
                avg_step = sum(logger.episode_step_logs)/len(logger.episode_step_logs)

                success_steps = []
                for i in range(len(logger.episode_success_logs)):
                    if logger.episode_success_logs[i]:
                        success_steps.append(logger.episode_step_logs[i])
                if len(success_steps) > 0:
                    avg_success_step = sum(success_steps) / len(success_steps)
                else:
                    avg_success_step = 1000

                result_file = os.path.join(logger.result_directory, "case" + str(case) + ".txt")
                with open(result_file, "w") as out_file:
                    out_file.write(
                        "%s %.18e %.18e %.18e %.18e\n"
                        % (
                            lang_goal,
                            avg_success,
                            avg_step,
                            avg_success_step,
                            avg_reward,
                        )
                    )
                case += 1
                print("\033[034m Language goal: {}, average steps: {}/{}, average reward: {}, average success: {}\033[0m".format(lang_goal, avg_step, avg_success_step, avg_reward, avg_success))
                logging.info(
                    f"Language goal: {lang_goal}, average steps: {avg_step}/{avg_success_step}, average reward: {avg_reward}, average success: {avg_success}")
                wandb.log({
                    "lang_goal": lang_goal,
                    "avg_success": avg_success,
                    "avg_step": avg_step,
                    "avg_success_step": avg_success_step,
                    "avg_reward": avg_reward
                })


    ray.shutdown()
