## CogniMatch
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

### Web Frontend / Backend

[KAILAS]

## Challenges we ran into

### Hallucinations and AI Overreach
One of the most difficult parts of this project was dealing with agents that ocassionally produced unsupported claims , hallucinated facts, or weird dialogues. While prompt-engineering and heuristic-based checks helped mitigate some of these issues, the introduction of a `Safety Agent` that leveraged a more comprehensive reasoning model to filter out responses that seemed too speculative or factually unsound. This allowed us to decrease unsafe messages and hallucinations.

### Photo Sending and Agents Interaction

Orchestrating multiple agents in real-time while ensuring each agent's role remained clear was difficult. It was difficult to handle multi modal data and to ensure relevant images were sent correctly during conversations. Using a conversation manager, prompt-engineering, and processing code to guide the LLMs helped us get around this. 

### Creating Detailed Yet Concise Profiles

One of our core goals was to transform raw survey data into an expressive but compact representation of a user's personality and communication style. First, we fetch responses via Google Forms, converting any uploads to Base64. Processed data is passed into a profile builder function (the Survey class) that uses `gemini-2.0-flash` to generate descriptions for the images, and we condense every user’s answers into a “profile matrix”. This matrix not only encodes the user’s conversational style but also avoids the clutter of raw survey text.

### Connecting Web Front End / Backend With AI Python Backend
Integrating a web frontend / backend with 3D animations and multiple Python AI endpoints on a cloud VM required a well planned data flow strategy / architecture. We established well-defined API routes and architecture in FastAPI and web sockets to stream conversations and images through the different techstack layers. 

## Accomplishments that we're proud of
- We successfully orchestrated four specialized agents using different Gemini model variants, all running in real-time and interacting meaningfully.
- Our system doesn't just handle text; agents can also exchange and discuss user-uploaded images

## What we learned
Implementing and using Google Cloud VM, FastAPI endpoints, websockets, Gemini API, Google Drive & Sheets API
