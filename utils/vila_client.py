import os
from typing import Any, Dict, List

import requests

VILA_BASE_URL = os.environ.get("FREEGRASP_VILA_BASE_URL", "http://127.0.0.1:8001")
VILA_MODEL = os.environ.get("FREEGRASP_VILA_MODEL", "Qwen")
VILA_TIMEOUT = float(os.environ.get("FREEGRASP_VILA_TIMEOUT", "600"))


class VilaServiceClient:
    def __init__(self, base_url: str, model: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False

    def _extract_text(self, payload: Dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            raise ValueError(f"Missing choices in VILA response: {payload}")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    return item.get("text", "").strip()
        raise ValueError(f"Unsupported VILA response format: {payload}")

    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        system_parts: List[str] = []
        normalized: List[Dict[str, Any]] = []

        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                if isinstance(content, str) and content.strip():
                    system_parts.append(content.strip())
                continue
            normalized.append(message)

        if not system_parts:
            return normalized

        system_text = "\n\n".join(system_parts).strip()
        if not normalized:
            return [{"role": "user", "content": system_text}]

        first = normalized[0]
        if first.get("role") != "user":
            normalized.insert(0, {"role": "user", "content": system_text})
            return normalized

        first_content = first.get("content", "")
        if isinstance(first_content, str):
            first["content"] = f"{system_text}\n\n{first_content}".strip()
            return normalized

        if isinstance(first_content, list):
            first["content"] = [{"type": "text", "text": system_text}] + first_content
            return normalized

        return normalized

    def generate(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
        top_p: float = 1.0,
        num_beams: int = 1,
        use_cache: bool = True,
    ) -> str:
        # Parse messages for Qwen format
        system_prompt = ""
        instruction = ""
        image_path = ""
        
        for msg in messages:
            if msg["role"] == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_prompt = content
                elif isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            system_prompt = item["text"]
            elif msg["role"] == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    instruction = content
                elif isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            instruction = item["text"]
                        elif item.get("type") == "image_url":
                            # Assume image is base64 or path, but Qwen needs base64
                            image_url = item["image_url"]
                            if isinstance(image_url, dict) and "url" in image_url:
                                url = image_url["url"]
                                if url.startswith("data:image"):
                                    image_b64 = url.split(",", 1)[1]
                                else:
                                    # Read file and encode
                                    with open(url, "rb") as f:
                                        image_b64 = base64.b64encode(f.read()).decode("utf-8")
                            else:
                                with open(str(image_url), "rb") as f:
                                    image_b64 = base64.b64encode(f.read()).decode("utf-8")
                            image_path = f"data:image/png;base64,{image_b64}"
        
        payload = {
            "system_prompt": system_prompt,
            "instruction": instruction,
            "image_path": image_path,
            "max_new_tokens": max_new_tokens
        }
        
        response = self.session.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        return result["text"]


reasoning_client = VilaServiceClient(VILA_BASE_URL, VILA_MODEL, VILA_TIMEOUT)
