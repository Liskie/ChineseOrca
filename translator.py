import json
import os
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Pool

import requests
from jsonlines import jsonlines
from tenacity import retry, wait_random_exponential, stop_after_attempt
from datasets import load_dataset
from tqdm import tqdm


class SupportedModels(Enum):
    GPT4 = "gpt-4"
    ChatGPT = "gpt-3.5-turbo-0613"


class SupportedDatasetTypes(Enum):
    Huggingface = 'huggingface'
    Local = 'local'


@dataclass
class LocaleData:
    system_prompt: str
    question: str
    response: str


@dataclass
class Datapoint:
    id: str
    en: LocaleData = None
    zh: LocaleData = None


class OrcaTranslator:

    def __init__(self, dataset_type=SupportedDatasetTypes.Huggingface):
        match dataset_type:
            case SupportedDatasetTypes.Huggingface:
                self.dataset = load_dataset('Open-Orca/OpenOrca')['train']
            case SupportedDatasetTypes.Local:
                self.dataset = load_dataset("parquet", data_files={
                    'GPT4': '/Users/Liskie/Projects/PycharmProjects/OpenOrca/1M-GPT4-Augmented.parquet',
                    'ChatGPT': '/Users/Liskie/Projects/PycharmProjects/OpenOrca/3_5M-GPT3_5-Augmented.parquet'
                })['GPT4']
            case _:
                raise ValueError(f'Invalid dataset type selected. Supported types are {list(SupportedDatasetTypes)}')

        self.id2datapoint: dict[str, Datapoint] = {}

        for line in self.dataset:
            self.id2datapoint[line['id']] = Datapoint(
                id=line['id'],
                en=LocaleData(
                    system_prompt=line['system_prompt'],
                    question=line['question'],
                    response=line['response']
                )
            )

    @staticmethod
    def dir_check(directory: str) -> None:
        # Check if the output directory exists
        if not os.path.exists(os.path.dirname(directory)):
            os.makedirs(os.path.dirname(directory))

    def translate_instructions(self, num_lines: int = None, num_workers: int = 4) -> None:
        # Validate num_lines
        if num_lines and num_lines > len(self.dataset):
            raise ValueError(f'num_lines={num_lines} is larger than the length of the dataset ({len(self.dataset)}).')

        # Distribute the work to multiple processes
        inputs = list(self.id2datapoint.values())[:num_lines] if num_lines else list(self.id2datapoint.values())
        with Pool(num_workers) as pool:
            modified_datapoints = tqdm(pool.imap(self.translate_question, inputs),
                                       total=len(inputs),
                                       desc='Translating instructions: ')

        # Update the datapoints
        for datapoint in modified_datapoints:
            self.id2datapoint[datapoint.id] = datapoint

        # Dump the datapoints
        self.dump_datapoints('output/datapoints_translation_only.jsonl')

    def translate_question(self, datapoint: Datapoint, model=SupportedModels.GPT4) -> Datapoint:
        translation = self.request_model(f'Please translate the following text into simplified Chinese:\n'
                                         f'{datapoint.en.question}', model)
        datapoint.zh.question = translation
        return datapoint

    def generate_answers(self, num_lines: int = None, num_workers: int = 1) -> None:
        # Validate num_lines
        if num_lines and num_lines > len(self.dataset):
            raise ValueError(f'num_lines={num_lines} is larger than the length of the dataset ({len(self.dataset)}).')

        # Distribute the work to multiple processes
        inputs = list(self.id2datapoint.values())[:num_lines] if num_lines else list(self.id2datapoint.values())
        with Pool(num_workers) as pool:
            modified_datapoints = tqdm(pool.imap(self.ask_question, inputs),
                                       total=len(inputs),
                                       desc='Generation responses: ')

        # Update the datapoints
        for datapoint in modified_datapoints:
            self.id2datapoint[datapoint.id] = datapoint

        # Dump the datapoints
        self.dump_datapoints('output/datapoints.jsonl')

    def ask_question(self, datapoint: Datapoint, model=SupportedModels.GPT4) -> Datapoint:
        response = self.request_model(f'Please answer the following question:\n{datapoint.zh.question}', model)
        datapoint.zh.response = response
        return datapoint

    def dump_datapoints(self, output_path: str = 'output/datapoints.jsonl') -> None:
        self.dir_check(output_path)
        with jsonlines.open(output_path, 'w') as writer:
            writer.write_all(self.id2datapoint)

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
            "model": str(model),
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
        print(json.dumps(response, indent=4, ensure_ascii=False))
        text = response["choices"][0]["message"]["content"].strip()
        return text
