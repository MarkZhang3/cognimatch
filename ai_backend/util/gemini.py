import os
import time
from typing import List, Union, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import google.generativeai as genai
import base64

# ------------------------
# Global Rate Limiting
# ------------------------
RATE_LIMIT = 500
requests_made_this_minute = 0
minute_start_time = time.time()

def check_rate_limit():
    """
    Naive bucket-based rate limiter. It uses a 60-second window:
      - If the limit is reached before the 60s are up, we sleep until the window resets.
      - Then we reset the counter and the start time.
    """
    global requests_made_this_minute, minute_start_time
    
    now = time.time()
    elapsed = now - minute_start_time
    
    # If more than 60 seconds have passed since last reset, start a new window
    if elapsed >= 60:
        requests_made_this_minute = 0
        minute_start_time = now
    
    # If we're at or above limit and still in the current window, sleep
    if requests_made_this_minute >= RATE_LIMIT:
        sleep_time = 60 - elapsed
        print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds...")
        time.sleep(sleep_time)
        # Reset counters
        requests_made_this_minute = 0
        minute_start_time = time.time()
    
    # Count the new request
    requests_made_this_minute += 1

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


@dataclass
class GeminiMultimodalRequest:
    parts: List[Union[str, GeminiImage]]


@dataclass
class GeminiResponse:
    text: str
    raw: Optional[dict] = None


# --- Gemini Handler ---
class GeminiHandler:
    def __init__(self, model_name: str="gemini-2.0-flash"):
        genai.configure(api_key=API_KEY)
        self.model = genai.GenerativeModel(model_name)

    def send_text_prompt(self, request: GeminiTextRequest) -> GeminiResponse:
        # Check rate limit before sending
        check_rate_limit()
        
        response = self.model.generate_content(request.prompt)
        return GeminiResponse(text=response.text, raw=response)

    def send_multimodal_prompt(self, request: GeminiMultimodalRequest) -> GeminiResponse:
        # Check rate limit before sending
        check_rate_limit()
        
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

        response = self.model.generate_content(parts)
        return GeminiResponse(text=response.text, raw=response)

    def send_multimodal_prompt_b64(
        self,
        prompt: str,
        b64_image_strs: list[str],
        mime_type: str = "image/png"
    ) -> GeminiResponse:
        """
        Accepts a list of base64-encoded image strings and sends a multimodal prompt.
        Strips data URI prefix if present in any image.
        """
        images = []
        for b64_image_str in b64_image_strs:
            # Strip data URI prefix if it's there
            if b64_image_str.startswith("data:"):
                b64_image_str = b64_image_str.split(",", 1)[1]
            try:
                image_bytes = base64.b64decode(b64_image_str)
            except Exception as e:
                raise ValueError("Failed to decode base64 image string") from e

            images.append(GeminiImage(mime_type=mime_type, data=image_bytes))

        # Include the prompt and all images in the parts list
        parts = [prompt] + images
        # print('images len:')
        # print(len(images))
        request = GeminiMultimodalRequest(parts=parts)
        return self.send_multimodal_prompt(request)