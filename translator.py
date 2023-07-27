import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from functools import partial
from multiprocessing import Pool
from typing import Generator

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


class SupportedModel(Enum):
    GPT4 = "gpt-4"
    ChatGPT = "gpt-3.5-turbo-0613"


class SupportedDatasetType(Enum):
    Huggingface = 'huggingface'
    LismbpLocal = 'local'
    HPCLocal = 'hpc_local'


@dataclass
class LocaleData:
    system_prompt: str
    question: str
    response: str

    def to_json(self) -> dict:
        return {
            'system_prompt': self.system_prompt,
            'question': self.question,
            'response': self.response
        }

    @classmethod
    def from_json(cls, json_data: dict) -> 'LocaleData':
        return cls(
            system_prompt=json_data['system_prompt'],
            question=json_data['question'],
            response=json_data['response']
        )


@dataclass
class Datapoint:
    id: str
    en: LocaleData | None = None
    zh: LocaleData | None = None

    def to_json(self) -> dict:
        return {
            'id': self.id,
            'en': self.en.to_json() if self.en else None,
            'zh': self.zh.to_json() if self.zh else None
        }

    @classmethod
    def from_json(cls, json_data: dict) -> 'Datapoint':
        return cls(
            id=json_data['id'],
            en=LocaleData.from_json(json_data['en']) if json_data['en'] else None,
            zh=LocaleData.from_json(json_data['zh']) if json_data['zh'] else None
        )


class DataBuffer:

    def __init__(self, size=100, dump_path='output/datapoints_translation_only.jsonl'):
        self.buffer: list[Datapoint] = []
        self.size = size
        self.dump_path = dump_path

        # Clear the dump file
        dir_check(dump_path)
        with open(dump_path, 'w') as _:
            pass

    def add(self, datapoint: Datapoint) -> None:
        self.buffer.append(datapoint)
        if len(self.buffer) >= self.size:
            self.dump()
            self.buffer = []

    def dump(self) -> None:
        logger.info(f'Dumping {len(self.buffer)} datapoints into {self.dump_path}.')
        with jsonlines.open(self.dump_path, 'a') as writer:
            writer.write_all([datapoint.to_json() for datapoint in self.buffer])


class OrcaTranslator:

    def __init__(self,
                 datapoint_file: str = 'output/datapoints_vanilla.jsonl',
                 dataset_type=SupportedDatasetType.Huggingface,
                 buffer_size=100,
                 num_datapoints_to_process=None,
                 num_workers=4):

        self.dataset = None
        self.datapoint_buffer: DataBuffer = DataBuffer(size=buffer_size)

        self.num_workers = num_workers
        self.num_processed_datapoints: int = 0

        self.datapoint_file = datapoint_file

        self.system_prompt_translations: list[dict[str, str]] = []

        # Count datapoints
        with open(datapoint_file, 'r') as reader:
            self.num_datapoints_total = sum(1 for _ in reader)

        self.num_datapoints_to_process = num_datapoints_to_process if num_datapoints_to_process else self.num_datapoints_total

        # Validate num_datapoints_to_process
        if self.num_datapoints_to_process > self.num_datapoints_total:
            raise ValueError(
                f'Number of datapoints to process ({self.num_datapoints_to_process}) is larger than the length of the '
                f'dataset ({self.num_datapoints_total}).')

        # Check if there exists prepared datapoints in the output directory
        if os.path.exists(datapoint_file):
            logger.info(f'Found prepared datapoints in {datapoint_file}.')
        else:
            logger.info(f'No prepared datapoints found.')
            self.load_dataset(dataset_type)
            self.generate_datapoints()

    @property
    def datapoints(self):
        return self.load_datapoints(self.datapoint_file)

    def load_datapoints(self, input_path: str) -> Generator[Datapoint, None, None]:
        logger.info(f'{self.num_datapoints_to_process} out of {self.num_datapoints_total} datapoints will be loaded '
                    f'from {input_path} in iterator mode.')
        with jsonlines.open(input_path, 'r') as reader:
            for i, datapoint in enumerate(reader):
                if i >= self.num_datapoints_to_process:
                    break
                yield Datapoint.from_json(datapoint)

    def load_dataset(self, dataset_type: SupportedDatasetType) -> None:
        logger.info(f'Loading dataset from {dataset_type}.')
        match dataset_type:
            case SupportedDatasetType.Huggingface:
                self.dataset = load_dataset('Open-Orca/OpenOrca')['train']
            case SupportedDatasetType.LismbpLocal:
                self.dataset = load_dataset("parquet", data_files={
                    'GPT4': '/Users/Liskie/Projects/PycharmProjects/OpenOrca/1M-GPT4-Augmented.parquet',
                    'ChatGPT': '/Users/Liskie/Projects/PycharmProjects/OpenOrca/3_5M-GPT3_5-Augmented.parquet'
                })['GPT4']
            case SupportedDatasetType.HPCLocal:
                self.dataset = load_dataset("parquet", data_files={
                    'GPT4': '/home/sytl8692/lispace/datasets/OpenOrca/1M-GPT4-Augmented.parquet',
                    'ChatGPT': '/home/sytl8692/lispace/datasets/OpenOrca/3_5M-GPT3_5-Augmented.parquet'
                })['GPT4']
            case _:
                raise ValueError(f'Invalid dataset type selected. Supported types are {list(SupportedDatasetType)}')
        logger.info(f'Loaded dataset with {len(self.dataset)} lines.')

    def generate_datapoints(self):
        logger.info(f'Generating datapoints.')
        datapoints: dict[str, Datapoint] = {}
        for line in tqdm(self.dataset, desc='Generating datapoints: '):
            datapoints[line['id']] = Datapoint(
                id=line['id'],
                en=LocaleData(
                    system_prompt=line['system_prompt'],
                    question=line['question'],
                    response=line['response']
                ),
                zh=LocaleData(
                    system_prompt='',
                    question='',
                    response=''
                )
            )
        dump_path = 'output/datapoints_vanilla.jsonl'
        logger.info(f'Generated {len(datapoints)} datapoints. Dumping datapoints into {dump_path}.')
        # self.dump_datapoints(dump_path)
        logger.info(f'Datapoints dumped.')

    def translate_system_prompts(self,
                                 system_prompt_cache_path='output/system_prompt_translations.jsonl',
                                 force_retranslate=False) -> None:
        dir_check(system_prompt_cache_path)
        if os.path.exists(system_prompt_cache_path) and not force_retranslate:
            logger.info(f'Found prepared system prompt translations in {system_prompt_cache_path}. Loading them.')
            with jsonlines.open(system_prompt_cache_path, 'r') as reader:
                for translation in reader:
                    self.system_prompt_translations.append(translation)
        else:
            logger.info(f'No prepared system prompt translations found. Collecting and translating them.')
            # Collect unique prompts
            unique_prompts = set(datapoint.en.system_prompt for datapoint in self.load_datapoints(self.datapoint_file))
            # Translate the prompts
            translate_system_prompt_gpt4 = partial(self.translate_system_prompt, model=SupportedModel.GPT4)
            with Pool(self.num_workers) as pool:
                with tqdm(total=len(unique_prompts), desc='Translating system prompts: ') as pbar:
                    for prompt_pair in pool.imap(translate_system_prompt_gpt4, unique_prompts):
                        pbar.update()
                        self.system_prompt_translations.append(prompt_pair)
            # Dump the translation pairs
            with jsonlines.open(system_prompt_cache_path, 'w') as writer:
                writer.write_all(self.system_prompt_translations)

    def translate_system_prompt(self, system_prompt: str, model: SupportedModel) -> dict[str, str]:
        logger.info(f'Process {os.getpid()} is translating system prompt {system_prompt}.')
        if not system_prompt:
            return {
                'en': '',
                'zh': ''
            }
        return {
            'en': system_prompt,
            'zh': self.request_model(f'Please translate the following text into Simplified Chinese:\n{system_prompt}',
                                      model)
        }

    def translate_instructions(self) -> None:
        # Distribute the work to multiple processes
        logger.info(f'Distributing work to {self.num_workers} workers.')

        translate_question_gpt4 = partial(self.translate_question, model=SupportedModel.GPT4)

        with Pool(self.num_workers) as pool:
            with tqdm(total=self.num_datapoints_to_process, desc='Translating instructions: ') as pbar:
                for datapoint in pool.imap(translate_question_gpt4, self.datapoints):
                    self.datapoint_buffer.add(datapoint)
                    pbar.update()

        # Dump the datapoints
        self.datapoint_buffer.dump()
        logger.info(f'Translation completed.')

    def translate_question(self, datapoint: Datapoint, model: SupportedModel) -> Datapoint:
        logger.info(f'Process {os.getpid()} is translating question for datapoint {datapoint.id}.')
        question = self.request_model(f'Please translate the following text into simplified Chinese:\n'
                                      f'{datapoint.en.question}', model)
        for pair in self.system_prompt_translations:
            if pair['en'] == datapoint.en.system_prompt:
                datapoint.zh.system_prompt = pair['zh']
                break
        datapoint.zh.question = question
        return datapoint

    # def generate_answers(self) -> None:
    #     # Distribute the work to multiple processes
    #     inputs = [(datapoint, SupportedModel.GPT4)
    #               for datapoint in (list(self.datapoints.values())[:num_lines] if num_lines
    #                                 else list(self.datapoints.values()))]
    #     with Pool(self.num_workers) as pool:
    #         modified_datapoints = tqdm(pool.istarmap(self.ask_question, inputs),
    #                                    total=len(inputs),
    #                                    desc='Generation responses: ')
    #
    #     # Update the datapoints
    #     for datapoint in modified_datapoints:
    #         self.datapoints[datapoint.id] = datapoint

    # Dump the datapoints
    # self.dump_datapoints('output/datapoints.jsonl')

    def ask_question(self, args) -> Datapoint:
        datapoint, model = args
        response = self.request_model(f'Please answer the following question:\n{datapoint.zh.question}', model)
        datapoint.zh.response = response
        return datapoint

    # def dump_datapoints(self, output_path: str = 'output/datapoints_vanilla.jsonl', mode='w') -> None:
    #     dir_check(output_path)
    #     with jsonlines.open(output_path, mode) as writer:
    #         writer.write_all([datapoint.to_json() for datapoint in self.datapoints.values()])

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def request_model(self, prompt: str, model=SupportedModel.GPT4) -> str:
        match model:
            case SupportedModel.GPT4:
                request_ip = "http://120.92.10.46:8080/chat"
            case SupportedModel.ChatGPT:
                request_ip = "http://47.254.22.102:8989/chat"
            case _:
                raise ValueError(f'Invalid model selected. Supported models are {list(SupportedModel)}.')

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
        # print('New translation: ', json.dumps(response, indent=4, ensure_ascii=False))
        if 'error' in response:
            return f"{response['error']['code']}: {response['error']['message']}"
        return response["choices"][0]["message"]["content"].strip()
