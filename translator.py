from enum import Enum

import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
from datasets import load_dataset
from tqdm import tqdm


class SupportedModels(Enum):
    GPT4 = "gpt-4"
    ChatGPT = "gpt-3.5-turbo-0613"


class OrcaTranslator:

    def __init__(self, dataset='Open-Orca/OpenOrca'):
        self.dataset = load_dataset(dataset)['train']
        self.instruction2answer: dict[str, str] = {}

    def translate_instructions(self, num_lines: int = None) -> None:
        for i, line in tqdm(enumerate(self.dataset)):
            if num_lines and i >= num_lines:
                break
            self.instruction2answer[self.translate_text(line['question'])] = ''

    def generate_answers(self, num_lines: int = None) -> None:
        for i, instruction in tqdm(enumerate(self.instruction2answer.keys())):
            if num_lines and i >= num_lines:
                break
            self.instruction2answer[instruction] = self.ask_question(instruction)

    def translate_text(self, text: str, model=SupportedModels.GPT4) -> str:
        return self.request_model(f'Please translate the following text into simplified Chinese:\n{text}', model)

    def ask_question(self, question: str, model=SupportedModels.GPT4) -> str:
        return self.request_model(f'Please answer the following question:\n{question}', model)

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(1))
    def request_model(self, prompt: str, model=SupportedModels.GPT4) -> str:
        match model:
            case SupportedModels.GPT4:
                request_ip = "http://120.92.10.46:8080/chat"
            case SupportedModels.ChatGPT:
                request_ip = "http://47.254.22.102:8989/chat"
            case _:
                raise ValueError(f'Invalid model selected. Supported models are {list(SupportedModels)}.')

        response = requests.post(request_ip, json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 1,
            "top_p": 1,
            "max_tokens": 2048,
            "presence_penalty": 0,
            "frequency_penalty": 0
        }).json()
        # print(json.dumps(response, indent=4, ensure_ascii=False))
        text = response["choices"][0]["message"]["content"].strip()
        return text

