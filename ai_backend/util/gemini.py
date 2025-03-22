import os
from typing import Callable, List, Union, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env file
load_dotenv()
API_KEY = os.getenv("API_KEY_GEMINI")
if not API_KEY:
    raise RuntimeError("Missing API_KEY_GEMINI in .env file")

# --- Data Structures ---
@dataclass
class GeminiImage:
    mime_type: str
    data: bytes


@dataclass
class GeminiTextRequest:
    prompt: str
    stream: bool = False
    on_stream: Optional[Callable[[str], None]] = None


@dataclass
class GeminiMultimodalRequest:
    parts: List[Union[str, GeminiImage]]
    stream: bool = False
    on_stream: Optional[Callable[[str], None]] = None


@dataclass
class GeminiResponse:
    text: str
    raw: Optional[dict] = None


# --- Gemini Handler ---
class GeminiHandler:
    def __init__(self):
        genai.configure(api_key=API_KEY)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def send_text_prompt(self, request: GeminiTextRequest) -> GeminiResponse:
        if request.stream and request.on_stream:
            stream = self.model.generate_content(request.prompt, stream=True)
            full_text = ""
            for chunk in stream:
                part = chunk.text
                if part:
                    request.on_stream(part)
                    full_text += part
            return GeminiResponse(text=full_text)
        else:
            response = self.model.generate_content(request.prompt)
            return GeminiResponse(text=response.text, raw=response)

    def send_multimodal_prompt(self, request: GeminiMultimodalRequest) -> GeminiResponse:
        parts = []
        for item in request.parts:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, GeminiImage):
                parts.append({
                    "mime_type": item.mime_type,
                    "data": item.data
                })
            else:
                raise ValueError("Unsupported input part: must be str or GeminiImage")

        if request.stream and request.on_stream:
            stream = self.model.generate_content(parts, stream=True)
            full_text = ""
            for chunk in stream:
                part = chunk.text
                if part:
                    request.on_stream(part)
                    full_text += part
            return GeminiResponse(text=full_text)
        else:
            response = self.model.generate_content(parts)
            return GeminiResponse(text=response.text, raw=response)
