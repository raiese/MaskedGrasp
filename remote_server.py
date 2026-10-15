import argparse
import base64
import inspect
import json
import os
import re
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

os.environ.setdefault("MPLBACKEND", "Agg")

from models.FGC_graspnet.utils.data_utils import CameraInfo
from run import compute_grasp_pose
from utils.config import get_args_parser


def _safe_request_id(raw_value):
    value = str(raw_value or "").strip()
    if not value:
        value = datetime.now().strftime("%Y%m%d_%H%M%S")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value[:128] or datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_text_if_exists(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _decode_base64_to_file(encoded, path):
    with open(path, "wb") as f:
        f.write(base64.b64decode(encoded))


def _make_camera_info(camera_intrinsics):
    return CameraInfo(
        width=int(camera_intrinsics["width"]),
        height=int(camera_intrinsics["height"]),
        fx=float(camera_intrinsics["fx"]),
        fy=float(camera_intrinsics["fy"]),
        cx=float(camera_intrinsics["cx"]),
        cy=float(camera_intrinsics["cy"]),
        scale=float(camera_intrinsics["scale"]),
    )


def _build_infer_args(server_args):
    parser = argparse.ArgumentParser(
        "FreeGraspRemoteCompat",
        parents=[get_args_parser()],
        add_help=False,
    )
    infer_args = parser.parse_args([])
    infer_args.checkpoint_grasp_path = server_args.checkpoint_grasp_path
    infer_args.viz = server_args.viz
    return infer_args


def _collect_artifacts(request_dir, artifact_names):
    artifacts = {}
    for name in artifact_names:
        path = os.path.join(request_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            artifacts[name] = base64.b64encode(f.read()).decode("ascii")
    return artifacts


def _prepare_request_dir(output_root, request_id):
    os.makedirs(output_root, exist_ok=True)
    request_dir = os.path.join(output_root, request_id)
    if os.path.exists(request_dir):
        request_dir = os.path.join(
            output_root,
            f"{request_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    os.makedirs(request_dir, exist_ok=True)
    return request_dir


def process_infer_request(payload, server_args):
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    files = payload.get("files")
    camera_intrinsics = payload.get("camera_intrinsics")
    task_text = str(payload.get("task", "")).strip()
    if not isinstance(files, dict):
        raise ValueError("Missing files object in request.")
    if not isinstance(camera_intrinsics, dict):
        raise ValueError("Missing camera_intrinsics object in request.")
    if not task_text:
        raise ValueError("task must not be empty.")

    image_b64 = files.get("image_png_base64")
    depth_b64 = files.get("depth_npz_base64")
    if not image_b64 or not depth_b64:
        raise ValueError("files must contain image_png_base64 and depth_npz_base64.")

    request_id = _safe_request_id(payload.get("request_id"))
    request_dir = _prepare_request_dir(server_args.output_root, request_id)

    request_meta = {
        "request_id": request_id,
        "capture_timestamp": payload.get("capture_timestamp"),
        "received_timestamp": datetime.now().isoformat(timespec="seconds"),
        "task": task_text,
        "camera_intrinsics": camera_intrinsics,
    }
    _write_json(os.path.join(request_dir, "request.json"), request_meta)
    _write_json(os.path.join(request_dir, "camera_intrinsics.json"), camera_intrinsics)
    with open(os.path.join(request_dir, "task.txt"), "w", encoding="utf-8") as f:
        f.write(task_text)
    _decode_base64_to_file(image_b64, os.path.join(request_dir, "image.png"))
    _decode_base64_to_file(depth_b64, os.path.join(request_dir, "depth.npz"))

    camera = _make_camera_info(camera_intrinsics)
    infer_args = _build_infer_args(server_args)
    compute_signature = inspect.signature(compute_grasp_pose)
    if len(compute_signature.parameters) >= 3:
        grasp_pose = compute_grasp_pose(request_dir, camera, infer_args)
    else:
        grasp_pose = compute_grasp_pose(request_dir, camera)

    response = {
        "request_id": request_id,
        "request_dir": request_dir,
        "log": _read_text_if_exists(os.path.join(request_dir, "log.txt")),
        "traceback": _read_text_if_exists(os.path.join(request_dir, "traceback.txt")),
    }

    if isinstance(grasp_pose, dict) and "error" in grasp_pose:
        response["ok"] = False
        response["error"] = str(grasp_pose["error"])
        if "exception_type" in grasp_pose:
            response["exception_type"] = str(grasp_pose["exception_type"])
        response["traceback"] = _read_text_if_exists(grasp_pose.get("traceback_path", "")) or response["traceback"]
    elif isinstance(grasp_pose, BaseException):
        response["ok"] = False
        response["error"] = str(grasp_pose)
        response["exception_type"] = type(grasp_pose).__name__
    else:
        response["ok"] = True
        response["grasp_pose"] = grasp_pose

    want_artifacts = bool(payload.get("return_artifacts", False))
    if want_artifacts:
        artifact_names = payload.get("artifact_names") or server_args.default_artifacts
        if isinstance(artifact_names, list):
            artifact_names = [str(name) for name in artifact_names if str(name).strip()]
        else:
            artifact_names = server_args.default_artifacts
        response["artifacts"] = _collect_artifacts(request_dir, artifact_names)

    _write_json(os.path.join(request_dir, "response.json"), response)
    return response


class RemoteHandler(BaseHTTPRequestHandler):
    server_version = "FreeGraspRemoteHTTP/0.1"

    def _send_json(self, data, status_code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/healthz":
            self._send_json({
                "ok": True,
                "service": "freegrasp-remote",
                "time": datetime.now().isoformat(timespec="seconds"),
            })
            return
        self._send_json({"ok": False, "error": "Not found"}, status_code=404)

    def do_POST(self):
        if self.path != "/infer_grasp":
            self._send_json({"ok": False, "error": "Not found"}, status_code=404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                raise ValueError("Empty request body.")
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))
            response = process_infer_request(payload, self.server.server_args)
            status = 200 if response.get("ok") else 500
            self._send_json(response, status_code=status)
        except Exception as exc:
            tb = traceback.format_exc()
            self._send_json(
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": tb,
                },
                status_code=500,
            )

    def log_message(self, fmt, *args):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {self.address_string()} {fmt % args}")


def build_parser():
    parser = argparse.ArgumentParser("FreeGrasp Remote Inference Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output_root", type=str, default="data/remote_requests")
    parser.add_argument("--checkpoint_grasp_path", type=str, default="logs/checkpoint_fgc.tar")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument(
        "--default_artifacts",
        nargs="*",
        default=["molmo_label.png", "molmo_id.txt", "log.txt", "grasp_pose.json"],
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), RemoteHandler)
    server.server_args = args
    print(f"Serving FreeGrasp remote inference on http://{args.host}:{args.port}")
    print(f"Saving requests under: {os.path.abspath(args.output_root)}")
    server.serve_forever()


if __name__ == "__main__":
    main()
