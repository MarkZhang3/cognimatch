from util.gemini import GeminiHandler, GeminiResponse, GeminiTextRequest, GeminiImage

SYSTEM_PROMPT = """
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

        # Buffer for partial text chunks during streaming
        self._partial_text_buffer = ""

    def _build_prompt_for_gemini(self) -> str:
        """
        Builds the text prompt to send to Gemini, including:
          - The system prompt
          - This agent's profile
          - The entire conversation history (in a format that does not leak "[YOU]")
        """
        # Start with the system prompt
        lines = "[SYSTEM_PROMPT]\n"
        lines += SYSTEM_PROMPT.strip() + "\n\n"

        # Add the agent's profile
        lines += "[PROFILE]\n"
        lines += self.profile.strip() + "\n\n"

        # Add the message history in a readable format
        lines += "[MESSAGE HISTORY]\n"
        for entry in self.message_log:
            frm = entry["from"]
            msg = entry["message"]
            lines += f"{frm}\n{msg}\n"

        return lines

    def _on_stream_callback(self, partial_text: str):
        """
        Callback for streaming partial text. 
        Prints as it arrives and accumulates it.
        """
        # Print partial text as it comes in
        print(partial_text, end="", flush=True)
        # Accumulate it for the final full response
        self._partial_text_buffer += partial_text

    def generate_response_streaming(self) -> str:
        """
        Fetches the next response from Gemini in streaming mode. 
        Prints text chunks as they arrive, then returns the full text.
        """
        # Clear the buffer for a new streaming response
        self._partial_text_buffer = ""

        prompt = self._build_prompt_for_gemini()
        request = GeminiTextRequest(
            prompt=prompt,
            stream=True,
            on_stream=self._on_stream_callback
        )

        self.gemini.send_text_prompt(request)

        # Move to a new line after streaming ends
        # print()
        return self._partial_text_buffer

    def talk_to(self, other_agent: "Agent", message: str):
        """
        Sends a message to another Agent (non-streaming).
        """
        # Log the message in this agent's history
        self.message_log.append({
            "from": self.name,
            "to": other_agent.name,
            "message": message
        })
        # print(f"\n{self.name} -> {other_agent.name}: {message}\n")

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
        # print(f"{self.name} received a message from {from_agent.name}: {message}")

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