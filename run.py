import os
os.environ.setdefault("MPLBACKEND", "Agg")
import cv2
import json
import argparse
import traceback
import numpy as np
import torch

from grasp_model import grasp_model
from models.langsam import langsamutils
from models.FGC_graspnet.utils.data_utils import CameraInfo

from utils.utils import *
from utils.config import *
from utils.graspnet_utils import get_correct_pose

# 初始化本地分割模型 (LangSAM 不需要联网)
# Uses the lazy shared instance from utils.config.

def compute_grasp_pose(path, camera_info, infer_args=None, manual_id=None, manual_class=None, manual_coor=None, manual_instance_id=None):
    if infer_args is None:
        parser = argparse.ArgumentParser('RUN', parents=[get_args_parser()])
        args = parser.parse_args([])
    else:
        args = infer_args
    
    try:
        path = os.path.abspath(str(path))
        image_path = os.path.join(path, "image.png")
        depth_path = os.path.join(path, "depth.npz")
        
        # 加载基础数据
        img_ori = cv2.imread(image_path)
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        depth_data = np.load(depth_path)
        depth_ori = depth_data['depth']
        image_pil = langsamutils.load_image(image_path)

        # --- 暴力修改：强制直接从 task.txt 读取结果，完全不请求服务器 ---
        task_path = os.path.join(path, "task.txt")
        
        # 逻辑：如果参数传了 manual_id 就用参数；否则尝试读 task.txt
        if manual_id is not None:
            goal = manual_class
            goal_id = manual_id
            goal_coor = manual_coor if manual_coor else [0, 0]
            print(f">>> [强制本地] 使用参数输入: {goal}")
        elif os.path.exists(task_path):
            # 如果文件存在，解析文件内容
            with open(task_path, 'r') as f:
                content = f.read()
                # 假设内容格式为 "target_id: 7, target_class: scissor"
                import re
                goal_id = re.search(r'target_id:\s*(\d+)', content).group(1)
                goal = re.search(r'target_class:\s*(\w+)', content).group(1)
                goal_coor = [0, 0] # 默认坐标
            print(f">>> [强制本地] 从 task.txt 读取目标: {goal}")
        else:
            raise Exception("错误：既没有 manual_id 也没有 task.txt，无法继续")
            
        # 记录本地运行日志
        with open(os.path.join(path, "log.txt"), "a") as file:
            file.write(f"\n[Execution] Target: id={goal_id}, class={goal}, coor={goal_coor}\n")

        # 3. --- 语义分割 (优先使用实例掩码；否则走 LangSAM) ---
        using_instance_mask = False
        if manual_instance_id is not None and 'instances_objects' in depth_data.files:
            instance_map = depth_data['instances_objects']
            goal_mask_np = (instance_map == int(manual_instance_id))
            if np.any(goal_mask_np):
                using_instance_mask = True
                ys, xs = np.where(goal_mask_np)
                goal_bbox = np.array(
                    [xs.min(), ys.min(), xs.max(), ys.max()],
                    dtype=np.float32
                )
                goal_mask = torch.from_numpy(goal_mask_np.astype(np.bool_))
                print(f">>> 使用 instances_objects 精确掩码: GT_ID={manual_instance_id}, 像素数={int(goal_mask_np.sum())}")
                cv2.imwrite(
                    os.path.join(path, "image_mask_1.png"),
                    (goal_mask_np.astype(np.uint8) * 255)
                )
            else:
                print(f">>> 警告: instances_objects 中找不到 GT_ID={manual_instance_id}，回退 LangSAM")

        if not using_instance_mask:
            print(f">>> 正在分割目标物体: {goal}...")
            masks, boxes, phrases, logits = langsam_actor.predict(image_pil, goal)

            if len(masks) == 0:
                print(f"FAILED: LangSAM 没能在图中找到 '{goal}'")
                return {"error": "Object not detected by LangSAM"}

            # 根据坐标匹配最接近的掩码
            goal_mask, mask_index = get_goal_mask_with_index(masks, goal_coor)
            goal_bbox = boxes[mask_index].cpu().numpy()
            langsam_actor.save(masks, boxes, phrases, logits, image_pil, path, viz=args.viz)

        # 生成裁剪区域
        cropping_box = create_cropping_box_from_boxes(goal_bbox, (img_ori.shape[1], img_ori.shape[0]))
        goal_mask_input = goal_mask.unsqueeze(0)

        # 可视化分割结果
        if args.viz:
            visualize_cropping_box(img_ori, cropping_box)
        
        # 4. --- 抓取姿态估计 (GraspNet 本地运行) ---
        print(">>> 正在生成抓取位姿...")
        endpoint, pcd = get_and_process_data(cropping_box, img_ori, depth_ori, camera_info, viz=args.viz)
        
        # Initialize GraspNet with runtime device fallback.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f">>> GraspNet device: {device}")
        g_model = grasp_model(args=args, device=device,
                             image=img_ori, mask=goal_mask_input, camera_info=camera_info)
            
        gg, _ = g_model.forward(endpoint, pcd, path)
        
        if len(gg) == 0:
            print("FAILED: GraspNet 未能生成有效位姿")
            return {"error": "No grasp poses generated"}
        else:
            # 获取最优抓取并保存
            R, t, w = get_correct_pose(gg[0], path, args.viz)
            data = {
                'translation': t.tolist(),
                'rotation': R.tolist(),
                'width': float(w)
            }
            
            save_path = os.path.join(path, "grasp_pose.json")
            with open(save_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
            print(f"SUCCESS: 结果已保存至 {save_path}")

        return data   

    except Exception as e:
        traceback_path = os.path.join(path, "traceback.txt")
        with open(traceback_path, "w", encoding="utf-8") as file:
            file.write(traceback.format_exc())
        print(f"CRITICAL ERROR: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    raise SystemExit("Use: python scripts/infer_grasp.py --help")
