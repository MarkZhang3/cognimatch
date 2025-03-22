from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

async def event_generator():
    # Simulate a stream of events (e.g. log lines, live updates, etc.)
    for i in range(1, 11):
        yield f"data: Message {i}\n\n"
        await asyncio.sleep(1)  # Simulate delay (e.g. real-time source)



@app.get("/save_form")
async def save_form(id: str, form: Dict):
    


@app.get("/stream")
async def stream():
    return StreamingResponse(event_generator(), media_type="text/event-stream")
