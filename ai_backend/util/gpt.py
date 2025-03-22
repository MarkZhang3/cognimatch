import os
import json
from enum import Enum
from dataclasses import dataclass
import requests
from dotenv import load_dotenv
import time
import tiktoken

HPC = False


load_dotenv()
API_KEY = os.getenv("API_KEY_OPENAI")
if not API_KEY:
    raise RuntimeError("Missing API_KEY_OPENAI in .env file")


# API
OPEN_AI_ENDPOINT = 'https://api.openai.com/v1/chat/completions'

# ENUMS
class ModelType(Enum):
    GPT_4O = 1
    GPT_3_TURBO = 2
    GPT_4O_MINI = 3
    GPT_O1 = 4


DEBUG = True

@dataclass
class ModelInfo:
    model_name: str
    cost_per_input_token: float
    cost_per_output_token: float
    encoding: tiktoken.Encoding

COST_PRESETS = {}
if not HPC:
    # PRESETS
    COST_PRESETS = {
        ModelType.GPT_4O:
            ModelInfo(
                model_name='gpt-4o',
                cost_per_input_token=0.000005,
                cost_per_output_token=0.000015,
                encoding=tiktoken.get_encoding('o200k_base')
            ),
        ModelType.GPT_3_TURBO:
            ModelInfo(
                model_name='gpt-3.5-turbo',
                cost_per_input_token=5e-7,
                cost_per_output_token=0.000002,
                encoding=tiktoken.get_encoding('cl100k_base')
            ),
        ModelType.GPT_4O_MINI:
            ModelInfo(
                model_name='gpt-4o-mini',
                cost_per_input_token=1.5e-7,
                cost_per_output_token=6e-7,
                encoding=tiktoken.get_encoding('o200k_base')
            ),
        ModelType.GPT_O1:
            ModelInfo(
                model_name="o1",
                cost_per_input_token=0.0000011,
                cost_per_output_token=0.0000044,
                encoding=tiktoken.get_encoding("o200k_base")
            ),
    }

class LLMModel:
    # for rate limiting
    _yield_per_call: int = 1
    def __init__(self, model_type: ModelType):
        self.type = model_type
        # if the name is already a cost preset
        self.model_cost_info = COST_PRESETS[model_type]
        # for general information
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        # for computing costs
        self.input_tokens_since_epoch: int = 0
        self.output_tokens_since_epoch: int = 0

    def get_cost_since_epoch(self) -> float:
        """Returns the cost of API calls since epoch in USD
        """
        if self.model_cost_info:
            return self.output_tokens_since_epoch * self.model_cost_info.cost_per_output_token + \
                   self.input_tokens_since_epoch * self.model_cost_info.cost_per_input_token
        return 0

    def get_total_cost(self) -> float:
        return self.total_input_tokens * self.model_cost_info.cost_per_input_token + \
               self.total_output_tokens * self.model_cost_info.cost_per_output_token

    def add_usage(self, input_tokens: int, output_tokens: int):
        """
        add tokens used to usage
        :param input_tokens: number of input tokens
        :param output_tokens: number of output tokens
        """
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.input_tokens_since_epoch += input_tokens
        self.output_tokens_since_epoch += output_tokens

    def new_epoch(self):
        """
        Starts a new period where tokens will be counted starting after this function call
        """
        self.input_tokens_since_epoch = 0
        self.output_tokens_since_epoch = 0


class LLM:
    _minute_start: float = 0
    _tokens_since_minute_start: int = 0
    _rate_limit_yield: int = 1
    _default_yield: int = 0.70
    _API_KEY = API_KEY
    # RATE LIMIT CONSTANTS
    TPM: int = 100000
    # yield lengths for error codes (in seconds)
    _NONE_200_YIELD: int = 20
    _429_YIELD: int = 120
    models = {
        ModelType.GPT_3_TURBO: LLMModel(ModelType.GPT_3_TURBO),
        ModelType.GPT_4O: LLMModel(ModelType.GPT_4O),
        ModelType.GPT_4O_MINI: LLMModel(ModelType.GPT_4O_MINI),
        ModelType.GPT_O1: LLMModel(ModelType.GPT_O1)
    }

    def __new__(cls, *args, **kwargs):
        raise TypeError(f"{cls.__name__} cannot be instantiated")


    @staticmethod
    def get_number_of_tokens(text: str, model_type: ModelType) -> int:
        return len(LLM.models[model_type].model_cost_info.encoding.encode(text))

    @staticmethod
    def can_message(system_prompt: str, user_message: str, model_type: ModelType) -> bool:
        """Returns true if an API call can be made under the rate limit"""
        num_tokens = len(LLM.models[model_type].model_cost_info.encoding.encode(system_prompt + user_message))
        if LLM._minute_start == 0:
            LLM._minute_start = time.perf_counter()
            return True
        if time.perf_counter() - LLM._minute_start > 60:
            # past a minute, so start a new period
            LLM._minute_start = time.perf_counter()
            LLM._tokens_since_minute_start = 0
        if LLM._tokens_since_minute_start + num_tokens > LLM.TPM * 0.9:
            # will get rate limited
            print('Rate Limited')
            return False
        return True

    @staticmethod
    def message(system_prompt: str, user_message: str, model_type: ModelType, temperature=0.5) -> str:
        """
        Attempts to message the specified LLM model type, yields if being rate limited
        :param system_prompt: the system prompt
        :param user_message: the user prompt
        :param model_type: the model type to be called
        :returns response message from GPT
        """
        time.sleep(LLM._default_yield)
        # yield until we can message again
        while not LLM.can_message(system_prompt, user_message, model_type):
            time.sleep(LLM._rate_limit_yield)
        # can call now
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
        if DEBUG:
            print('== SENDING TO: %s ==\n[System Prompt]\n%s\n[Message]\n%s\n===========' %
                       (LLM.models[model_type].model_cost_info.model_name, system_prompt, user_message))
        # send message
        request_dump = {}
        if model_type == ModelType.GPT_O1:
            # 1o requires special request
            request_dump = {
                "model": LLM.models[model_type].model_cost_info.model_name,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "System Prompt:\n" + system_prompt + '\n' + 'User Message:\n' + user_message}]}]
            }
        else:
            request_dump = {
                "model": LLM.models[model_type].model_cost_info.model_name,
                "messages": messages,
                "temperature": temperature,
                'max_completion_tokens': 100000
            }
        response = requests.post(
            url=OPEN_AI_ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + LLM._API_KEY
            },
            data=json.dumps(request_dump)
        )
        if response.status_code == 429:
            # being rate limited
            LLM._minute_start = time.perf_counter() + LLM._429_YIELD
            LLM._tokens_since_minute_start = LLM.TPM * LLM._429_YIELD / 60
            print('GPT CODE 429 | Retrying...')
            return LLM.message(system_prompt, user_message, model_type)
        if response.status_code != 200:
            LLM._minute_start = time.perf_counter() + LLM._NONE_200_YIELD
            LLM._tokens_since_minute_start = LLM.TPM * LLM._NONE_200_YIELD / 60
            print('GPT CODE %d | Retrying...' % response.status_code)
            print(response)
            return LLM.message(system_prompt, user_message, model_type)
        # successful call
        response_content = response.json()
        # calculate usage
        if 'usage' in response_content:
            completion_tokens = response_content['usage']
            input_tokens = completion_tokens['prompt_tokens']
            output_tokens = completion_tokens['completion_tokens']
            LLM._tokens_since_minute_start += completion_tokens['total_tokens']
            LLM.models[model_type].add_usage(input_tokens, output_tokens)
        # get response
        if 'choices' in response_content:
            messages = response_content['choices']
            if messages and 'message' in messages[0]:
                response_text = messages[0]['message']['content']
                if DEBUG:
                    print('== GPT: %s ==\n[Output]\n%s\n===========' %
                               (LLM.models[model_type].model_cost_info.model_name, response_text))
                return response_text
        # something went wrong
        print('GPT | No response')
        exit(1)

    @staticmethod
    def new_epoch():
        """
        Starts a new period where tokens will be counted starting after this function call
        """
        for model in LLM.models:
            LLM.models[model].new_epoch()

    @staticmethod
    def get_total_cost() -> float:
        """
        Gets the total cost made to the API
        :return: cost in USD
        """
        return sum([LLM.models[model].get_total_cost() for model in LLM.models])

    @staticmethod
    def get_total_cost_since_epoch() -> float:
        """
        Returns the total cost made to API since epoch
        :return: cost in USD
        """
        return sum([LLM.models[model].get_cost_since_epoch() for model in LLM.models])