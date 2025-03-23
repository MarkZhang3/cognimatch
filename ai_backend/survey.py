from util.gpt import LLM, ModelType
from util.gemini import GeminiHandler


SYSTEM_PROMPT = """
[SYSTEM PROMPT]
You are an advanced psychological profiler. You transform user-provided JSON survey data (and any accompanying text, including chat logs) into a structured json format describing that individual’s personality, communication style, and other background attributes about them.
Your output should be enough for someone to emulate a conversation for that person that is indistinguishable from the actual person.
Your objectives and requirements:
Data Extraction and Analysis
Receive JSON data and optional text (e.g., chat logs, free-text responses).
Parse and interpret the data for each relevant dimension of the individual’s personality and communication style.
Personality Dimensions
Identify and score traits such as (but not limited to):
- Extraversion vs. Introversion
- Agreeableness vs. Challenging Demeanor
- Conscientiousness vs. Carelessness
- Emotional Range (e.g., calm vs. reactive)
- Openness (e.g., creativity, curiosity)
- Communication Style (e.g., formality, directness, humor usage, empathy)
- Tone and Demeanor (e.g., upbeat, stern, polite, sarcastic)
- Interests / hobbies
- Place of birth
- Etc...
You may also include additional subtleties gleaned from text, such as:
- Typical sentence length and complexity
- Lexical variety and specific word choices
- Use of hedging or disclaimers
- Level of emotional expression
- Degree of positivity, negativity, or neutrality

Output: Json dictionary

Produce a structured json dictionary listing each attribute (row) along with:

A brief label or trait name (e.g. “Extraversion”)

A concise description of what that trait entails (e.g. “Energy from social interaction, tendency to be outgoing”).

A numeric or qualitative rating (or both) reflecting the individual’s standing on that trait, based on the parsed data.

Any sub-traits or distinguishing qualities if relevant (e.g., “Prefers small gatherings but uses expressive language”).

If the data is insufficient to assess a trait, mark it with an indication such as “Insufficient Data.”

Your output should focus on these structured traits and descriptors; avoid extraneous content.

Important Constraints & Style
- Do not reveal raw survey responses or chat logs verbatim in the final matrix. Only provide the synthesized personality/behavioral inferences.
- If asked for disclaimers or references, you may omit them.
- Assume you are not a licensed mental-health provider—your task is strictly to produce a structured summarization (not a clinical diagnosis).
- Keep your final output to a concise, clear, and well-organized matrix or table.

Instructions Recap
Read and parse: Incorporate all relevant JSON fields and textual elements.

Analyze: Infer core personality traits and communication patterns from the data.

Output: Present a json output composing everything about that individual. Make the output long, capture everything.
[BELOW ARE THE SURVEY RESULTS]
"""
IMAGE_CAPTIONER_SYSTEM_PROMPT = """
You are a image captioner. Given an image, you must only output what is being shown in the image, nothing else. Only output relvant details and nothing else, try to be very detailed like explaining everything in the image but concise.
"""


image_captioner = GeminiHandler("gemini-2.0-flash")


class Survey:
    def __init__(self, agent_id, results: dict):
        self.agent_id = agent_id
        self.results = results
        self.images = []
        self.image_captions = []
        self.user_descriptions = []
        # remove b64 images
        if "Pictures (base64)" in self.results:
            # get the images
            self.images = self.results["Pictures (base64)"]
            self.user_descriptions = self.results["Captions"]
            # remove them from the dict
            del results["Captions"]
            del results["Pictures (base64)"]
        if len(self.images) > 0:
            # there's images so caption all of them
            for b64_image in self.images:
                image_caption = image_captioner.send_multimodal_prompt_b64(IMAGE_CAPTIONER_SYSTEM_PROMPT, [b64_image]).text
                print(image_caption)
                self.image_captions.append(image_caption)
        self.results = str(results)
        # build avail_images
        self.avail_images = {}
        for i in range(len(self.images)):
            self.avail_images[f"image_{i}"] = {
                "automated_caption": self.image_captions[i],
                "user_description": self.user_descriptions[i],
                "b64": self.images[i]
            }
        # get the json results
        self.profile = LLM.message(SYSTEM_PROMPT, self.results, ModelType.GPT_O1)

    def get_profile_matrix(self)->dict:
        return self.profile

    def get_images_as_str(self)->str:
        s = ""
        for i in range(len(self.images)):
            key = f"image_{i}"
            s += f"{key}: (user description: {self.avail_images[key]['user_description']}, automated caption: {self.avail_images[key]['automated_caption']})\n"
        s += '\n'
        return s


if __name__ == '__main__':
    # test
    results_test = """
    {
  "metadata": {
    "survey_id": "personality_survey_999",
    "timestamp": "2025-03-22T20:45:00Z",
    "version": "2.0"
  },
  "respondent_info": {
    "age": 42,
    "gender": "Non-binary",
    "occupation": "Research Scientist",
    "education_level": "Ph.D.",
    "country_of_residence": "Canada"
  },
  "personality_traits": {
    "big_five": {
      "openness": 9,
      "conscientiousness": 5,
      "extraversion": 6,
      "agreeableness": 7,
      "neuroticism": 3
    },
    "sub_traits": {
      "anxiety": 3,
      "altruism": 8,
      "assertiveness": 5,
      "imagination": 9,
      "intellect": 9,
      "orderliness": 4,
      "excitement_seeking": 6,
      "emotionality": 4,
      "diligence": 6,
      "patience": 7
    },
    "additional_personality_insights": {
      "risk_tolerance": 7,
      "impulsivity": 4,
      "adaptability": 8
    }
  },
  "communication_style": {
    "formality": 6,
    "directness": 5,
    "humor_usage": 5,
    "empathy": 8,
    "verbosity": 6,
    "active_listening": 7,
    "punctuality_in_responses": 5,
    "preference_for_written_vs_spoken": "Written"
  },
  "preferences": {
    "conversation_topics": [
      "Advanced Physics",
      "Climate Change",
      "Philosophy of Science",
      "Hiking & Outdoor Activities",
      "Global Travel"
    ],
    "avoid_topics": [
      "Reality TV Shows",
      "Gossip",
      "Office Politics"
    ],
    "collaboration_style": "Prefers smaller group discussions over large meetings",
    "work_environment": [
      "Quiet workplaces with minimal interruptions",
      "Flexible schedules",
      "Remote collaboration options"
    ],
    "learning_methods": [
      "Reading research papers",
      "Hands-on experiments",
      "Attending seminars and conferences"
    ]
  },
  "open_ended_responses": [
    {
      "question_id": "Q1",
      "question_text": "What motivates you most in your professional life?",
      "response": "I’m driven by intellectual curiosity and the desire to solve complex problems that can have a real positive impact on society."
    },
    {
      "question_id": "Q2",
      "question_text": "How do you usually handle disagreements in a team?",
      "response": "I encourage open dialogue, prefer direct communication of issues, and seek compromises based on objective data where possible."
    },
    {
      "question_id": "Q3",
      "question_text": "Describe your ideal work environment.",
      "response": "A setting that balances individual focus time with collaborative brainstorming sessions, where new ideas are freely exchanged, and everyone feels heard."
    },
    {
      "question_id": "Q4",
      "question_text": "What personal values shape your approach to work?",
      "response": "Integrity, transparency, and respect for diverse viewpoints are the cornerstones of how I engage with colleagues."
    },
    {
      "question_id": "Q5",
      "question_text": "When under stress, how do you typically cope?",
      "response": "I step away briefly to refocus, maybe do some mindfulness exercises, then return to address the problem methodically."
    },
    {
      "question_id": "Q6",
      "question_text": "How would you describe your decision-making process?",
      "response": "Evidence-based, but I also consider the human element. Data is crucial, yet I’m aware that people’s feelings and perspectives matter."
    },
    {
      "question_id": "Q7",
      "question_text": "If you could improve one aspect of your communication, what would it be?",
      "response": "I’d like to be more succinct. Sometimes I delve too deeply into details and lose my audience."
    },
    {
      "question_id": "Q8",
      "question_text": "How do you define success in your projects or collaborations?",
      "response": "Success means measurable results and outcomes that benefit the stakeholders, plus a growth experience for everyone involved."
    },
    {
      "question_id": "Q9",
      "question_text": "Describe a situation where you had to adapt your communication style.",
      "response": "Working with a cross-cultural team required me to slow down, be mindful of language barriers, and ensure everyone had the chance to speak."
    },
    {
      "question_id": "Q10",
      "question_text": "What is your biggest strength when working in a team?",
      "response": "I foster synergy by helping bridge knowledge gaps, ensuring each member’s expertise is utilized effectively."
    }
  ]
}
    """
    s = Survey(1, results_test)
    print(s.get_profile_matrix())