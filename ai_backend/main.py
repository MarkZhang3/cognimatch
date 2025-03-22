from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.exceptions import HTTPException
from agent import Agent, SafetyAgent, EvaluatorAgent
from util.gemini import GeminiHandler
import asyncio
from pydantic import BaseModel
from survey import Survey
from typing import Dict, Any


app = FastAPI()

surveys: dict[str, Survey] = {}

async def event_generator():
    # Simulate a stream of events (e.g. log lines, live updates, etc.)
    for i in range(1, 11):
        yield f"data: Message {i}\n\n"
        await asyncio.sleep(1)  # Simulate delay (e.g. real-time source)


class SaveFormRequest(BaseModel):
    id: str
    form: Dict[str, Any]  


class StartConvoRequest(BaseModel):
    convo_id: str
    speaker_1_id: str
    speaker_2_id: str

@app.post("/save_form")
async def save_form_for_user(data: SaveFormRequest):
    if not data.id in surveys:
        if not data.form:
            return {"status": "failed", "reason": "Form is empty"}
        surveys[data.id] = Survey(data.id, data.form)
        return {"status": "sucess", "id": data.id}

# def start_convo_simulation(agent1, agent2, )

@app.post("/start_convo")
async def start_conversation(data: StartConvoRequest):
    speaker_1_id = data.speaker_1_id
    speaker_2_id = data.speaker_2_id
    if speaker_1_id not in surveys:
        raise HTTPException(status_code=400, detail=f"Speaker 1 (ID: {speaker_1_id}) has not saved the survey yet.")
    
    if speaker_2_id not in surveys:
        raise HTTPException(status_code=400, detail=f"Speaker 2 (ID: {speaker_2_id}) has not saved the survey yet.")

    # both have their profiles saved so now start conversation
    # create gemini handlers
    agent_gemini_handler = GeminiHandler(model_name="gemini-2.0-flash")
    reasoning_gemini_handler = GeminiHandler(model_name="gemini-1.5-pro")
    quick_gemini_handler = GeminiHandler(model_name="gemini-1.5-flash-8b")
    # build agents
    agent_1 = Agent(speaker_1_id, surveys[speaker_1_id].get_profile_matrix(), agent_gemini_handler)
    agent_2 = Agent(speaker_2_id, surveys[speaker_2_id].get_profile_matrix(), agent_gemini_handler)
    # build safety agent
    safety_agent = Agent(f"safety_{speaker_1_id}_{spekear_2_id}", agent_1, agent_2, reasoning_gemini_handler)
    # build evaluator
    evaluator_agent = EvaluatorAgent(agent_1, agent_2, reasoning_gemini_handler)
    # 

@app.get("/stream")
async def stream():
    return StreamingResponse(event_generator(), media_type="text/event-stream")
