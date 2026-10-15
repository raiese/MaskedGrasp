import argparse
import base64
import os
import traceback
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any, List, Literal, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
except ImportError as exc:
    raise ImportError(
        "Qwen2.5-VL requires a recent transformers build. "
        "Install with: pip install git+https://github.com/huggingface/transformers accelerate"
    ) from exc


IMAGE_CONTENT_BASE64_PREFIXES = ("data:image/png;base64,", "data:image/jpeg;base64,", "data:image/jpg;base64,")


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageURL(BaseModel):
    url: str


class ImageContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURL


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[Union[TextContent, ImageContent]]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 256
    top_p: Optional[float] = 1.0
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False
    use_cache: Optional[bool] = True
    num_beams: Optional[int] = 1


model = None
processor = None


def decode_base64_image(image_url: str) -> Image.Image:
    for prefix in IMAGE_CONTENT_BASE64_PREFIXES:
        if image_url.startswith(prefix):
            payload = image_url[len(prefix):]
            return Image.open(BytesIO(base64.b64decode(payload))).convert("RGB")
    raise ValueError(f"Unsupported image payload: {image_url[:32]}...")


def normalize_messages(messages: List[ChatMessage]) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    qwen_messages: list[dict[str, Any]] = []
    images: list[Image.Image] = []
    pending_system: list[str] = []

    for message in messages:
        if message.role == "system":
            if isinstance(message.content, str) and message.content.strip():
                pending_system.append(message.content.strip())
            continue

        content_items: list[dict[str, Any]] = []
        if isinstance(message.content, str):
            merged = "\n\n".join(pending_system + [message.content.strip()]).strip() if pending_system else message.content.strip()
            if merged:
                content_items.append({"type": "text", "text": merged})
            pending_system = []
        else:
            if pending_system:
                content_items.append({"type": "text", "text": "\n\n".join(pending_system).strip()})
                pending_system = []
            for item in message.content:
                if item.type == "text":
                    content_items.append({"type": "text", "text": item.text})
                elif item.type == "image_url":
                    image = decode_base64_image(item.image_url.url)
                    images.append(image)
                    content_items.append({"type": "image", "image": image})

        qwen_messages.append({"role": message.role, "content": content_items})

    if pending_system:
        qwen_messages.insert(0, {"role": "user", "content": [{"type": "text", "text": "\n\n".join(pending_system).strip()}]})

    return qwen_messages, images


def build_request_from_payload(payload: dict[str, Any]) -> ChatCompletionRequest:
    messages: list[ChatMessage] = []

    system_prompt = payload.get("system_prompt", "")
    if isinstance(system_prompt, str) and system_prompt.strip():
        messages.append(ChatMessage(role="system", content=system_prompt))

    instruction = payload.get("instruction", "")
    image_path = payload.get("image_path", "")

    content_items: list[Union[TextContent, ImageContent]] = []
    if isinstance(instruction, str) and instruction.strip():
        content_items.append(TextContent(type="text", text=instruction))
    if isinstance(image_path, str) and image_path.strip():
        content_items.append(ImageContent(type="image_url", image_url=ImageURL(url=image_path)))

    if content_items:
        messages.append(ChatMessage(role="user", content=content_items))

    return ChatCompletionRequest(
        model=payload.get("model", "qwen2.5-vl"),
        messages=messages,
        max_tokens=int(payload.get("max_new_tokens", payload.get("max_tokens", 256))),
        top_p=float(payload.get("top_p", 1.0)),
        temperature=float(payload.get("temperature", 0.0)),
        stream=bool(payload.get("stream", False)),
        use_cache=bool(payload.get("use_cache", True)),
        num_beams=int(payload.get("num_beams", 1)),
    )


def generate_text(request: ChatCompletionRequest) -> str:
    qwen_messages, images = normalize_messages(request.messages)
    text = processor.apply_chat_template(qwen_messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=images if images else None,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens or 256,
            do_sample=(request.temperature or 0.0) > 0,
            temperature=request.temperature or 0.0,
            top_p=request.top_p or 1.0,
            num_beams=request.num_beams or 1,
            use_cache=request.use_cache if request.use_cache is not None else True,
        )

    trimmed_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text.strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor
    model_path = app.args.model_path
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print(f"Model {model_path} loaded successfully.")
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        output_text = generate_text(request)
        return {
            "id": uuid.uuid4().hex,
            "object": "chat.completion",
            "created": time.time(),
            "model": request.model,
            "choices": [{"message": {"role": "assistant", "content": output_text}}],
        }
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/generate")
async def generate(payload: dict[str, Any]):
    try:
        request = build_request_from_payload(payload)
        output_text = generate_text(request)
        return {"text": output_text}
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    host = os.getenv("FREEGRASP_QWEN_HOST", "127.0.0.1")
    port = int(os.getenv("FREEGRASP_QWEN_PORT", "8001"))
    model_path = os.getenv("FREEGRASP_QWEN_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=host)
    parser.add_argument("--port", type=int, default=port)
    parser.add_argument("--model-path", type=str, default=model_path)
    app.args = parser.parse_args()

    uvicorn.run(app, host=app.args.host, port=app.args.port, workers=1)
