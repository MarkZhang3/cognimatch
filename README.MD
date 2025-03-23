# CogniMatch

## Inspiration
For some, the thought of starting a conversation with "Let's start with an icebreaker" or "So, what do you do?" can feel dull or even anxiety-inducing. These boring openers often don't capture who we truly are or what we even care about. 

We wondered: *why not personalize our conversations* based on deeper AI-driven insights-real personality, interests, speaking styles, and more? 

## What it does

By building agents that reflect our unique personality, attributes, and hobbies and then letting these agents interact in a controlled virtual environment, we can watch and tailor potential conversations even before they happen in the real world. 

This approach has the potential to uplift matchmaking, social conversations, and even dating. Instead of relying on surface-level algorithms or arbitrary elo systems, we can create interactions that respects real personalities and create more authentic  data-driven experiences. It's no longer simply a **"swipe-left-or-right"** interaction, but now a deeper exploration on compatibility based on how we may actually conversate with each other. By simulating these interactions, we can see firsthand where conversations flow naturally-and where they might get bumpy-giving us deep insights into who we actually connect with and why. 

## How we built it
### Cloud:

Our AI API backend is hosted on a Ubuntu-based virtual machine using `Google Cloud's Compute Engine` service. Specifically, the agent API was written entirely in Python and utilized FastAPI for all its endpoints. It also interfaced with many LLM providers i.e Google Gemini and OpenAI's chatGPT with a balanced choice of different models depending on the task. 

### Building Agents (Profile)
**Survey**

To get started, users complete a custom-designed survey which is similar to a MBTI survey but with more open-ended / purposeful questions to extract all the information required to create an agentic representation of their conversational-habits.  

**Building Profile From Survey**

Using heuristic-based processes and preprocessing, we first filter and prepare the completed survey for the reasoning model (`chatGPT o1`). Afterwards, we use prompt-engineering and postprocessing to create an object representation (profile) for the individual, the agent will use this profile to guide its conversations. 

### Agents
We utilized four specialized agents—Conversation, Safety, Evaluator, and Sentiment—each calling a different Gemini model variant. 

Our agents were created by using `chatGPT-o1` and other preprocessing techniques to create a profile of the user based on the results of the survey they've filled out at the beginning. 

#### Conversation Agent
Uses `Gemini 2.0 Flash` for generating context-rich, dialog-style responses. This endpoint supports multimodal (text+image) prompts, so the agent can send and receive images from each other. This model was our model of choice as the huge one million context window allowed it to keep track of its user's generated profile, conversation history, previously sent images, and worked well for the task. 

#### Safety Agent
Uses `Gemini 1.5 Pro` for deeper, policy-driven reasoning. Before or during chat, it checks if messages are safe to continue or if they violate safety guidelines.

#### Evaluator Agent
Also uses `Gemini 1.5 Pro` to analyze conversation logs to produce compatibility scores between speakers and in-depth analysis of the conversation. 

#### Sentiment Agent
Uses `Gemini 1.5 Flash (8B)` (lightweight but powerful) to classify user utterance in real time-determining if the agent is engaged, bored, or excited based on the given user profile data too.

## Challenges we ran into

### Hallucinations and AI Overreach
One of the most difficult parts of this project was dealing with agents that ocassionally produced unsupported claims , hallucinated facts, or weird dialogues. While prompt-engineering and heuristic-based checks helped mitigate some of these issues, the introduction of a `Safety Agent` that leveraged a more comprehensive reasoning model to filter out responses that seemed too speculative or factually unsound. This allowed us to decrease unsafe messages and hallucinations.

### Photo Sending and Agents Interaction

Orchestrating multiple agents in real-time while ensuring each agent's role remained clear was difficult. It was difficult to handle multi modal data and to ensure relevant images were sent correctly during conversations. Using a conversation manager, prompt-engineering, and processing code to guide the LLMs helped us get around this. 

### Creating Detailed Yet Concise Profiles

One of our core goals was to transform raw survey data into an expressive but compact representation of a user's personality and communication style. First, we fetch responses via Google Forms, converting any uploads to Base64. Processed data is passed into a profile builder function (the Survey class) that uses `gemini-2.0-flash` to generate descriptions for the images, and we condense every user’s answers into a “profile matrix”. This matrix not only encodes the user’s conversational style but also avoids the clutter of raw survey text.

### Connecting Web Front End / Backend With AI Python Backend
Integrating a web frontend / backend with 3D animations and multiple Python AI endpoints on a cloud VM required a well planned data flow strategy / architecture. We established well-defined API routes and architecture in FastAPI and web sockets to stream conversations and images through the different techstack layers. 

## Instructions to run
0. Have users fill out the google form [link](https://docs.google.com/forms/d/e/1FAIpQLSd29u5Azwebf3KWatSvqHK5SJ-eNMvG6q3imLG2gZgb8n7lnA/viewform?usp=dialog)
1. Install requirements.txt, run `source env/bin/activate`
2. run `fetch_form_responses.py` to fetch data from forms 
3. run `uvicorn main:app --host 0.0.0.0 --port 8000 --reload` to run python api backend 

## Accomplishments that we're proud of
- We successfully orchestrated four specialized agents using different Gemini model variants, all running in real-time and interacting meaningfully.
- Our system doesn't just handle text; agents can also exchange and discuss user-uploaded images

## What we learned
Implementing and using Google Cloud VM, FastAPI endpoints, websockets, Gemini API, Google Drive & Sheets API
