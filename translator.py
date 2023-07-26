import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Pool

import requests
from jsonlines import jsonlines
from tenacity import retry, wait_random_exponential, stop_after_attempt
from datasets import load_dataset
from tqdm import tqdm


def dir_check(directory: str) -> None:
    # Check if the output directory exists
    if not os.path.exists(os.path.dirname(directory)):
        os.makedirs(os.path.dirname(directory))


log_path = 'logs/translator.log'
dir_check(log_path)
logging.basicConfig(filename=log_path,
                    level=logging.INFO,
                    format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger()


class SupportedModels(Enum):
    GPT4 = "gpt-4"
    ChatGPT = "gpt-3.5-turbo-0613"


class SupportedDatasetTypes(Enum):
    Huggingface = 'huggingface'
    LismbpLocal = 'local'
    HPCLocal = 'hpc_local'


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
        logger.info(f'Loading dataset from {dataset_type}.')
        match dataset_type:
            case SupportedDatasetTypes.Huggingface:
                self.dataset = load_dataset('Open-Orca/OpenOrca')['train']
            case SupportedDatasetTypes.LismbpLocal:
                self.dataset = load_dataset("parquet", data_files={
                    'GPT4': '/Users/Liskie/Projects/PycharmProjects/OpenOrca/1M-GPT4-Augmented.parquet',
                    'ChatGPT': '/Users/Liskie/Projects/PycharmProjects/OpenOrca/3_5M-GPT3_5-Augmented.parquet'
                })['GPT4']
            case SupportedDatasetTypes.HPCLocal:
                self.dataset = load_dataset("parquet", data_files={
                    'GPT4': '/home/sytl8692/lispace/datasets/OpenOrca/1M-GPT4-Augmented.parquet',
                    'ChatGPT': '/home/sytl8692/lispace/datasets/OpenOrca/3_5M-GPT3_5-Augmented.parquet'
                })['GPT4']
            case _:
                raise ValueError(f'Invalid dataset type selected. Supported types are {list(SupportedDatasetTypes)}')
        logger.info(f'Loaded dataset with {len(self.dataset)} lines.')

        logger.info(f'Creating datapoints.')
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
        logger.info(f'Created {len(self.id2datapoint)} datapoints.')

    def translate_instructions(self, num_lines: int = None, num_workers: int = 4) -> None:
        # Validate num_lines
        if num_lines and num_lines > len(self.dataset):
            raise ValueError(f'num_lines={num_lines} is larger than the length of the dataset ({len(self.dataset)}).')

        # Distribute the work to multiple processes
        logger.info(f'Distributing work to {num_workers} workers.')
        inputs = [(datapoint, SupportedModels.GPT4)
                  for datapoint in (list(self.id2datapoint.values())[:num_lines] if num_lines
                                    else list(self.id2datapoint.values()))]
        with Pool(num_workers) as pool:
            modified_datapoints = tqdm(pool.imap(self.translate_question, inputs),
                                       total=len(inputs),
                                       desc='Translating instructions: ')

        # Update the datapoints
        for datapoint in modified_datapoints:
            self.id2datapoint[datapoint.id] = datapoint

        # Dump the datapoints
        dump_path = 'output/datapoints_translation_only.jsonl'
        logger.info(f'Translation completed. Dumping datapoints into {dump_path}')
        self.dump_datapoints(dump_path)
        logger.info(f'Datapoints dumped.')

    def translate_question(self, args) -> Datapoint:
        datapoint, model = args
        logger.info(f'Process {os.getpid()} is translating question for datapoint {datapoint.id}.')
        translation = self.request_model(f'Please translate the following text into simplified Chinese:\n'
                                         f'{datapoint.en.question}', model)
        datapoint.zh = LocaleData(
            system_prompt='',
            question=translation,
            response=''
        )
        return datapoint

    def generate_answers(self, num_lines: int = None, num_workers: int = 1) -> None:
        # Validate num_lines
        if num_lines and num_lines > len(self.dataset):
            raise ValueError(f'num_lines={num_lines} is larger than the length of the dataset ({len(self.dataset)}).')

        # Distribute the work to multiple processes
        inputs = [(datapoint, SupportedModels.GPT4)
                  for datapoint in (list(self.id2datapoint.values())[:num_lines] if num_lines
                                    else list(self.id2datapoint.values()))]
        with Pool(num_workers) as pool:
            modified_datapoints = tqdm(pool.istarmap(self.ask_question, inputs),
                                       total=len(inputs),
                                       desc='Generation responses: ')

        # Update the datapoints
        for datapoint in modified_datapoints:
            self.id2datapoint[datapoint.id] = datapoint

        # Dump the datapoints
        self.dump_datapoints('output/datapoints.jsonl')

    def ask_question(self, args) -> Datapoint:
        datapoint, model = args
        response = self.request_model(f'Please answer the following question:\n{datapoint.zh.question}', model)
        datapoint.zh.response = response
        return datapoint

    def dump_datapoints(self, output_path: str = 'output/datapoints.jsonl') -> None:
        dir_check(output_path)
        with jsonlines.open(output_path, 'w') as writer:
            writer.write_all(self.id2datapoint)

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
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
