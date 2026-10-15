"""Single-frame grasp geometry from an explicitly supplied target (not the full paper pipeline)."""
import argparse
import json
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--camera-json', type=Path, required=True)
    parser.add_argument('--target-id', type=int, required=True)
    parser.add_argument('--target-class', required=True)
    parser.add_argument('--target-xy', nargs=2, type=int, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    for p in [args.input_dir/'image.png', args.input_dir/'depth.npz', args.camera_json, args.checkpoint]:
        if not p.is_file():
            parser.error(f'Missing input: {p}')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run import compute_grasp_pose
    from models.FGC_graspnet.utils.data_utils import CameraInfo
    from utils.config import get_args_parser
    camera = CameraInfo(**json.loads(args.camera_json.read_text()))
    infer_args = get_args_parser().parse_args([])
    infer_args.checkpoint_grasp_path = str(args.checkpoint)
    result = compute_grasp_pose(args.input_dir, camera, infer_args=infer_args,
                                manual_id=args.target_id, manual_class=args.target_class,
                                manual_coor=args.target_xy)
    print(json.dumps(result, indent=2))
    if 'error' in result:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
