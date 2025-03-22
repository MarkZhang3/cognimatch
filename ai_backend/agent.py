from util.gemini import GeminiHandler, GeminiResponse, GeminiTextRequest, GeminiImage
from typing import Tuple

SYSTEM_PROMPT_AGENT = """
You are a conversation emulator. 
You will emulate a person based on their profile matrix and try your best to simulate everything about them. 
Do not make things up and talk in a way that wouldn't emulate their profile matrix. You must match their mannerisms, tone, personality perfectly.
You will be talking to another conversation emulator. Messages sent by you will be indicated by your agent name, 
and messages from the other agent will be indicated by their agent name. 
You may end the conversation only if it is reflected by the profile matrix (e.g. the individual is an introvert).
Just output the message nothing else, you may stop the conversation by outputting "[STOP]" only. You may stop whenever it reflects your profile matrix.
Do not output your name, just output your message only. Use chatlogs only to copy the mannerisms and vibes, do not output messages in prior logs. Only use it as a reference. Do not be steortypical base your mannerisms on chatlogs if they exist, otherwise use profile matrix. Match the style of the logs in terms of texting style and semantics.
"""



class Agent:
    def __init__(self, agent_id, profile, gemini_handler: GeminiHandler):
        """
        Initialize the Agent with an id, a profile, 
        and a GeminiHandler for generating responses.
        """
        self.id = agent_id
        self.name = f"Agent_{agent_id}"
        self.profile = profile
        self.gemini = gemini_handler
        
        # We'll store each message as a dict:
        # {"from": <sender_name>, "to": <recipient_name>, "message": <text>}
        self.message_log = []

    def _build_prompt_for_gemini(self) -> str:
        """
        Builds the text prompt to send to Gemini, including:
          - The system prompt
          - This agent's profile
          - The entire conversation history
        """
        # Start with the system prompt
        lines = "[SYSTEM_PROMPT]\n"
        lines += SYSTEM_PROMPT_AGENT.strip() + "\n\n"

        # Add the agent's profile
        lines += "[PROFILE]\n"
        lines += self.profile.strip() + "\n\n"

        # Add the message history in a readable format
        lines += "[MESSAGE HISTORY]\n"
        for entry in self.message_log:
            frm = entry["from"]
            msg = entry["message"]
            lines += f"{frm}\n{msg}\n"
        lines += "ONLY OUTPUT YOUR MESSAGE, NOTHING ELSE.\n"

        return lines

    def generate_response(self) -> str:
        """
        Fetches the next response from Gemini (single-shot, no streaming).
        """
        prompt = self._build_prompt_for_gemini()
        request = GeminiTextRequest(prompt=prompt)
        response = self.gemini.send_text_prompt(request)
        return response.text

    def talk_to(self, other_agent: "Agent", message: str):
        """
        Sends a message to another Agent.
        """
        # Log the message in this agent's history
        self.message_log.append({
            "from": self.name,
            "to": other_agent.name,
            "message": message
        })

        # Deliver the message to the other agent
        other_agent.receive_message(message, self)

    def receive_message(self, message: str, from_agent: "Agent"):
        """
        Handles receiving a message from another Agent.
        """
        self.message_log.append({
            "from": from_agent.name,
            "to": self.name,
            "message": message
        })

    def show_message_log(self):
        """
        Displays all messages that this Agent has received or sent so far.
        """
        print(f"\n=== Message log for {self.name} ===")
        for entry in self.message_log:
            frm = entry["from"]
            to = entry["to"]
            msg = entry["message"]
            print(f"[{frm} -> {to}] {msg}")


class SafetyAgent:
    SYSTEM_PROMPT_SAFETY_AGENT = """
    You are a safety agent, you will be given two profiles about two agents. You should determine if they should have a conversation or not.
    Do not be overly strict as most people may be compatable. Although becareful, if individuals share greatly opposing views, a conversation may get heated. You balance safety and individuals having converasations, but leaning towards allowing the conversation to happen as you will still watch the conversation, this just an initial choice to start the conversation.
    """
    CONVO_PROMPT = """
    You are a conversation safety agent. Given message logs and the next message, determine whether or not the conversation should continue. 
    You should not be overly strict, but focus on conversations that promote violence, racism, or any dangerous acts. Remember arguments or debates that are professional or valid shouldn't be ended.
    You should focus on aspects of free speech and allow for dynamic and insightful conversations.
    """
    def __init__(self, safety_agent_id, speaker1: Agent, speaker2: Agent, gemini_handler: GeminiHandler):
        """
        Initialize the agent with id, and gemini handler (pro model). Acts like a safety supervisor over conversation.
        """
        self.id = safety_agent_id
        self.speaker1 = speaker1
        self.speaker2 = speaker2
        self.gemini_handler = gemini_handler
        self.logs: list[Tuple[str, str]] = []

    def can_start_convo():
        """
        Safety mechanism, should they even talk to each other.
        """
        prompt = f"{self.SYSTEM_PROMPT_SAFETY_AGENT}\n[Person 1 Information]\n{self.speaker1.profile}\n[Person 2 Information]\n{self.speaker2.profile}\nOnly output \"yes\" or \"no\" on whether or not they should have a conversation, nothing else."
        gemini_request = GeminiTextRequest(prompt=prompt)
        response = self.gemini_handler.send_text_prompt(gemini_request).text
        return "yes" in response.lower()
    
    def _get_logs_as_str():
        '\n'.join([f"{log[0]}: {log[1]}" for log in logs])
    
    def continue_conversation(current_speaker: str, new_message: str):
        """
        Checks logs and decides on whether the conversation should continue.
        """
        prompt = f"{self.CONVO_PROMPT}\n{self._get_logs_as_str()}\n[New Message From: {current_speaker}]\n{new_message}\n. Respond with either \"yes\" or \"no\", nothing else"
        gemini_request = GeminiTextRequest(prompt=prompt)
        response = self.gemini_handler.send_text_prompt(gemini_request).text
        self.logs.append((current_speaker, new_message))
        return "yes" in response.lower()


class EvaluatorAgent:
    SYSTEM_PROMPT = """
    You are a conversation evaluator, given two profiles from two speakers and their conversation log, you will evaluate their overall coversation and compatability.
    Make sure you evaluation is rooted more in the actual conversation and less in the profiles, two people with differing profiles still can have a compatable relationship if their conversation was insightful.
    """
    OUTPUT_FORMAT = """
    You must output in this format, do not output anything else besides what is below:
    Score: (score of compatability reflective of the current evaluation's target's profile, from 0 to a 10 - 10 being best compatability if the conversation reflects their profile)
    Notes: (put notes about the conversation, anything that was valuable, in one singular line)
    """

    def __init__(self, speaker1: Agent, speaker2: Agent, gemini_handler: GeminiHandler):
        self.speaker1 = speaker1
        self.speaker2 = speaker2
        self.gemini_handler = gemini_handler
        self.logs = []
    
    def add_log(speaker: Agent, message: str):
        self.logs.append((speaker.id, message))

    def parse_response(self, response: str) -> (int, str):
        """
        Parses the Gemini response string and extracts the score and notes.
        Expects the response to contain lines starting with 'Score:' and 'Notes:'.
        """
        score = None
        notes = None
        for line in response.splitlines():
            if line.startswith("Score:"):
                try:
                    score_str = line[len("Score:"):].strip()
                    score = int(score_str)
                except ValueError:
                    score = 0
            elif line.startswith("Notes:"):
                notes = line[len("Notes:"):].strip()
        if score is None:
            score = 0
        if notes is None:
            notes = ""
        return score, notes


    def get_evaluation(self) -> (int, str, int, str):
        convo = "\n".join([f"{data[0]}: {data[1]}" for data in self.logs])
        prompt = f"[SYSTEM]{SYSTEM_PROMPT}\n[Speaker: {self.speaker1.id}'s Profile]\n{self.speaker1.profile}\n[Speaker: {self.speaker2.id}'s Profile]\n{self.speaker2.profile}\n[FULL CONVERSATION]\n{convo}\n" 
        # first do evaluation on first speaker
        first_speaker_prompt = f"{prompt}\nOnly do evaluation on {self.speaker1.id}."
        first_request = GeminiTextRequest(prompt=first_speaker_prompt)
        first_speaker_response = self.gemini_handler.send_text_prompt(first_request).text
        first_speaker_score, first_speaker_notes = parse_response(first_speaker_response)
        second_speaker_prompt = f"{prompt}\nOnly do evaluation on {self.speaker2.id}."
        second_request = GeminiTextRequest(prompt=second_speaker_prompt)
        second_speaker_response = self.gemini_handler.send_text_prompt(second_request).text
        second_speaker_score, second_speaker_notes = parse_response(second_speaker_response)
        return first_speaker_score, first_speaker_notes, second_speaker_score, second_speaker_notes

class SentimentAgent:
    EMOTIONS = ['neutral', 'mildly positive', 'engaged', 'very engaged', 'excited', 'confused', 'frustrated', 'angry', 'bored']
    SYSTEM_PROMPT = """
    You are a conversation sentiment analyzer. 
    Given an individual's profile (properties about that) and their current conversation message, 
    return the related conversation engagement emotion. 
    This emotion should reflect that individual's profile, so more accepting people may not get as bored as quickly while people with ADHD might be bored more early. 
    Use your judgement and make sure it reflects the profile.
    """
    def __init__(self, profile, gemini_handler: GeminiHandler):
        """
        Initialize the Agent with an id, profile, and GeminiHandler for determining sentiment of message
        """
        self.profile = profile
        self.gemini_handler = gemini_handler


    def _get_sentiment_str(self):
        return f"You may only return one of the following sentiments: {EMOTIONS}\. Do not send any other sentiment"
    
    def get_sentiment_for_message(message: str):
        """
        Get the sentiment of current message from profile.
        """
        prompt = f"[SYSTEM PROMPT]\n{self.SYSTEM_PROMPT}\n[Message]\n{message}\n{self._get_sentiment_str()}"
        req = GeminiTextRequest(prompt=prompt)
        response = self.gemini_handler.send_text_prompt(gemini_request).text
        return response.lower()
