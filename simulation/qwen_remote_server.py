import os
os.environ['TMPDIR'] = '/tmp'
os.environ['TEMP'] = '/tmp'
os.environ['TMP'] = '/tmp'
os.environ['TEMPDIR'] = '/tmp'
os.environ['XDG_CACHE_HOME'] = '/tmp'
os.environ['HF_HOME'] = '/tmp'
os.environ['TRANSFORMERS_CACHE'] = '/tmp'

import argparse
import base64
import io
import json
import logging
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
except ImportError:
    try:
        from transformers import Qwen2VLForConditionalGeneration as QwenVLModel
    except ImportError:
        from transformers import AutoModelForVision2Seq as QwenVLModel


class QwenInferenceServer:
    def __init__(self, model_path):
        import tempfile
        import os
        os.environ['TMPDIR'] = '/tmp'
        tempfile.tempdir = '/tmp'
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = QwenVLModel.from_pretrained(
            model_path,
            torch_dtype=model_dtype,
            device_map="cuda",
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def generate(self, system_prompt, instruction, image_path, max_new_tokens):
        import tempfile
        import os
        os.environ['TMPDIR'] = '/tmp'
        tempfile.tempdir = '/tmp'
        print(f"Received image_path: {image_path[:50]}...")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
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

        if image_path.startswith("data:image"):
            # Base64
            header, encoded = image_path.split(",", 1)
            image_data = base64.b64decode(encoded)
            image_inputs = [Image.open(io.BytesIO(image_data)).convert("RGB")]
        else:
            image_inputs = [Image.open(image_path).convert("RGB")]
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
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


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def build_handler(inference_server):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/generate":
                self._write_json(404, {"error": f"Unsupported path: {self.path}"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                print("Received request", flush=True)
                text = inference_server.generate(
                    system_prompt=payload["system_prompt"],
                    instruction=payload["instruction"],
                    image_path=payload["image_path"],
                    max_new_tokens=int(payload.get("max_new_tokens", 768)),
                )
                self._write_json(200, {"text": text})
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                print(f"Error in handler: {exc}", flush=True)
                print(f"Traceback: {tb}", flush=True)
                logging.exception("Qwen request failed")
                self._write_json(500, {
                    "error": str(exc),
                    "traceback": tb,
                })

        def do_GET(self):
            if self.path == "/health":
                self._write_json(200, {"status": "ok"})
                return
            self._write_json(404, {"error": f"Unsupported path: {self.path}"})

        def log_message(self, format, *args):
            return

        def _write_json(self, status_code, payload):
            body = json.dumps(payload).encode("utf-8")
            try:
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                logging.warning("Client disconnected before response could be written.")
                return

    return Handler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument(
        "--model-path",
        default="../weights/Qwen2.5-VL-7B-Instruct",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    inference_server = QwenInferenceServer(args.model_path)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(inference_server))
    print(f"Qwen server listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
