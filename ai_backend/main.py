import os
import pickle
import json
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.exceptions import HTTPException
from agent import Agent, SafetyAgent, EvaluatorAgent, SentimentAgent
from util.gemini import GeminiHandler
import asyncio
from pydantic import BaseModel
from survey import Survey
from typing import Dict, Any



FORMS_DIR = "./database/forms"

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
    # Basic validation
    if not data.form:
        return {"status": "failed", "reason": "Form is empty"}

    # Construct file path, e.g. "forms/{id}.pkl"
    file_path = os.path.join(FORMS_DIR, f"{data.id}.pkl")

    # Check if a file with the same ID already exists
    if os.path.exists(file_path):
        return {"status": "failed", "reason": "Form with this ID already exists"}

    try:
        # Create your Survey object
        survey_obj = Survey(data.id, data.form)

        # Pickle the object to disk
        with open(file_path, "wb") as f:
            pickle.dump(survey_obj, f)

        # You may also keep a reference in a dictionary
        surveys[data.id] = survey_obj

    except Exception as e:
        return {"status": "failed", "reason": f"Could not save form: {str(e)}"}

    # Return success
    return {"status": "success", "id": data.id}
# def start_convo_simulation(agent1, agent2, )


def get_response_detailed(agent, response):
    message = response["text"]
    image_b64 = ""
    image_str = ""
    if response["image"] != "":
        # there's an image in the response
        # fetch details about the image
        image_details = agent.survey.avail_images[response["image"]]
        image_b64 = image_details["b64"]
        user_description = image_details["user_description"]
        image_caption = image_details["automated_caption"]
        return message, image_b64, f"(user description: {user_description}, image caption: {image_caption})"
    return message, "", ""
def start_convo(agent1: Agent, agent2: Agent, safety_agent: SafetyAgent, eval_agent: EvaluatorAgent, sentiment_agent_1: SentimentAgent, sentiment_agent_2: SentimentAgent, max_turns: int = 25, delay: float = 2.0):
    """
    Lets agent1 and agent2 talk to each other in a loop, 
    streaming each response in real-time, until one outputs "[STOP]" 
    or we hit max_turns of back-and-forth.
    A small delay can be introduced between messages using the 'delay' parameter.
    """
    # Start with agent1 greeting agent2
    agent1.talk_to(agent2, "[SYSTEM]\n YOU WILL BEGIN THE CONVERSATION AND START FIRST.")

    turn_count = 0
    while turn_count < max_turns:
        turn_count += 1
        print(f"\n--- Turn {turn_count} ({agent2.name} responding) ---")

        # Agent2 streams its response
        response2 = agent2.generate_response()
        text_2, image_b64_2, image_str_2 = get_response_detailed(agent2, response2)
        if "[STOP]" in text_2:
            print("\nAgent2 indicated stop.\n")
            break
        agent2.talk_to(agent1, text_2, image_b64_2, image_str_2)
        # Introduce a small delay
        time.sleep(delay)

        turn_count += 1
        if turn_count > max_turns:
            break

        print(f"\n--- Turn {turn_count} ({agent1.name} responding) ---")

        # Agent1 streams its response
        response1 = agent1.generate_response()
        text_1, image_b64_1, image_str_1 = get_response_detailed(agent1, response1)
        if "[STOP]" in text_1:
            print("\nAgent1 indicated stop.\n")
            break

        agent1.talk_to(agent2, text_1, image_b64_1, image_str_1)

        # Introduce a small delay
        time.sleep(delay)

    agent1.show_message_log()
    agent2.show_message_log()

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
    agent_1 = Agent(speaker_1_id, surveys[speaker_1_id], agent_gemini_handler)
    agent_2 = Agent(speaker_2_id, surveys[speaker_2_id], agent_gemini_handler)
    # build safety agent
    safety_agent = SafetyAgent(f"safety_{speaker_1_id}_{speaker_2_id}", agent_1, agent_2, reasoning_gemini_handler)
    # build sentiment
    sentiment_agent_1 = SentimentAgent(surveys[speaker_1_id].get_profile_matrix(), quick_gemini_handler)
    sentiment_agent_2 = SentimentAgent(surveys[speaker_2_id].get_profile_matrix(), quick_gemini_handler)
    # build evaluator
    evaluator_agent = EvaluatorAgent(agent_1, agent_2, reasoning_gemini_handler)
    # start the convo
    start_convo(agent_1, agent_2, safety_agent, evaluator_agent, sentiment_agent_1, sentiment_agent_2)

@app.get("/stream")
async def stream():
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.on_event("startup")
def load_surveys_from_disk():
    print("Loading surveys from disk...")
    for filename in os.listdir(FORMS_DIR):
        if filename.endswith(".pkl"):
            path = os.path.join(FORMS_DIR, filename)
            try:
                with open(path, "rb") as f:
                    survey_obj = pickle.load(f)  # Unpickle the Survey object
                form_id = filename.removesuffix(".pkl")  # Derive ID from filename
                surveys[form_id] = survey_obj
                print(f"Loaded survey: {form_id}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    print(f"Loaded {len(surveys)} surveys total.")