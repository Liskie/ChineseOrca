import json
import logging
import os
import re
from functools import partial
from multiprocessing import Pool
from typing import Generator, Optional

import requests
import yaml
from datasets import load_dataset
from jsonlines import jsonlines
from tenacity import retry, wait_random_exponential, retry_if_result, stop_after_attempt, wait_fixed, before_log
from tqdm import tqdm

from utils.functions import dir_check, retry_condition, contains_only_zh_en_num_punct, count_existing_datapoints
from utils.data_buffer import DataBuffer, Datapoint
from utils.data_structure import SupportedMode, SupportedDatasetType, SupportedRunPhase, SupportedModel, LocaleData


class OrcaTranslator:

    def __init__(self,
                 path_config_file: str = 'config/path.yaml',
                 prompt_config_file: Optional[str] = None,
                 datapoint_vanilla_path: Optional[str] = None,
                 datapoint_translation_only_path: Optional[str] = None,
                 datapoint_complete_path: Optional[str] = None,
                 log_path: Optional[str] = None,
                 mode: SupportedMode = SupportedMode.Continue,
                 dataset_type: SupportedDatasetType = SupportedDatasetType.Huggingface,
                 buffer_size=10,
                 num_datapoints_to_process=None,
                 num_workers=4,
                 ):
        # Prefer to use direct constructor arguments over those in the path_config_file
        if os.path.exists(path_config_file):
            with open(path_config_file) as reader:
                path_config = yaml.load(reader, Loader=yaml.FullLoader)

            if not prompt_config_file:
                prompt_config_file = path_config['config_paths']['prompt_config_file']
            if not datapoint_vanilla_path:
                datapoint_vanilla_path = path_config['data_paths']['datapoint_vanilla_path']
            if not datapoint_translation_only_path:
                datapoint_translation_only_path = path_config['data_paths']['datapoint_translation_only_path']
            if not datapoint_complete_path:
                datapoint_complete_path = path_config['data_paths']['datapoint_complete_path']
            if not log_path:
                log_path = path_config['project_paths']['log_path']

        dir_check(log_path)
        logging.basicConfig(filename=log_path,
                            level=logging.INFO,
                            format='[%(asctime)s] [%(levelname)s] %(message)s')
        self.logger = logging.getLogger()

        self.logger.info('OrcaTranslator started.')

        # Parse strings to enum objects
        mode = SupportedMode(mode)
        dataset_type = SupportedDatasetType(dataset_type)

        self.logger.info(f'Parsing arguments: \n\t'
                         f'path_config_file: {path_config_file}\n\t'
                         f'prompt_config_file: {prompt_config_file}\n\t'
                         f'datapoint_vanilla_path: {datapoint_vanilla_path}\n\t'
                         f'datapoint_translation_only_path: {datapoint_translation_only_path}\n\t'
                         f'datapoint_complete_path: {datapoint_complete_path}\n\t'
                         f'log_path: {log_path}\n\t'
                         f'mode: {mode}\n\t'
                         f'dataset_type: {dataset_type}\n\t'
                         f'buffer_size: {buffer_size}\n\t'
                         f'num_datapoints_to_process: {num_datapoints_to_process}\n\t'
                         f'num_workers: {num_workers}\n\t')

        self.mode = mode

        self.dataset = None
        self.buffer_size = buffer_size

        self.num_workers = num_workers

        self.datapoint_vanilla_path = datapoint_vanilla_path
        self.datapoint_translation_only_path = datapoint_translation_only_path
        self.datapoint_complete_path = datapoint_complete_path

        with open(prompt_config_file) as reader:
            self.prompt_config = yaml.load(reader, Loader=yaml.FullLoader)

        self.system_prompt_translations: list[dict[str, str]] = []

        # Count datapoints
        with open(datapoint_vanilla_path, 'r') as reader:
            self.num_datapoints_total = sum(1 for _ in reader)

        self.num_datapoints_to_process = num_datapoints_to_process if num_datapoints_to_process else self.num_datapoints_total

        # Validate num_datapoints_to_process
        if self.num_datapoints_to_process > self.num_datapoints_total:
            raise ValueError(
                f'Number of datapoints to process ({self.num_datapoints_to_process}) is larger than the length of the '
                f'dataset ({self.num_datapoints_total}).')

        # Check if there exists prepared datapoints in the output directory
        if os.path.exists(datapoint_vanilla_path):
            self.logger.info(f'Found prepared datapoints in {datapoint_vanilla_path}.')
        else:
            self.logger.info(f'No prepared datapoints found.')
            self.load_dataset(dataset_type)
            self.generate_datapoints()

    def load_datapoints(self, load_phase: SupportedRunPhase) -> Generator[Datapoint, None, None]:
        match load_phase:
            case SupportedRunPhase.SystemPromptTranslation:
                input_path = self.datapoint_vanilla_path
                output_path = None
            case SupportedRunPhase.QuestionTranslation:
                input_path = self.datapoint_vanilla_path
                output_path = self.datapoint_translation_only_path
            case SupportedRunPhase.ResponseGeneration:
                input_path = self.datapoint_translation_only_path
                output_path = self.datapoint_complete_path
            case _:
                raise ValueError(f'Unsupported load phase: {load_phase}. '
                                 f'All supported load phases are: {list(SupportedRunPhase)}')

        # Check how may datapoints has been previously processed and dumped
        if self.mode == SupportedMode.Continue and os.path.exists(output_path):
            with jsonlines.open(output_path, 'r') as reader:
                existing_datapoints = sum(1 for _ in reader)
        else:
            existing_datapoints = 0

        if self.mode == SupportedMode.Continue and existing_datapoints == 0:
            self.logger.warning(f'Starting from datapoint #1 in Continue mode.')

        self.logger.info(
            f'Datapoints #{existing_datapoints + 1} ~ #{existing_datapoints + self.num_datapoints_to_process} '
            f'of a total of {self.num_datapoints_total} datapoints will be loaded '
            f'from {input_path} in iterator mode.')

        with jsonlines.open(input_path, 'r') as reader:
            for i, datapoint in enumerate(reader):
                # Skip the existing datapoints
                if i < existing_datapoints:
                    continue
                # Stop when the number of datapoints to process is reached
                if i >= self.num_datapoints_to_process + existing_datapoints:
                    break
                yield Datapoint.from_json(datapoint)

    def load_dataset(self, dataset_type: SupportedDatasetType) -> None:
        self.logger.info(f'Loading dataset from {dataset_type}.')
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
        self.logger.info(f'Loaded dataset with {len(self.dataset)} lines.')

    def generate_datapoints(self):
        self.logger.info(f'Generating datapoints.')
        datapoints: list[Datapoint] = []
        for line in tqdm(self.dataset, desc='Generating datapoints: '):
            datapoints.append(Datapoint(
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
            ))
        self.logger.info(
            f'Generated {len(datapoints)} datapoints. Dumping datapoints into {self.datapoint_vanilla_path}.')
        dir_check(self.datapoint_vanilla_path)
        with jsonlines.open(self.datapoint_vanilla_path, 'w') as writer:
            writer.write_all([datapoint.to_json() for datapoint in datapoints])
        self.logger.info(f'Datapoints dumped.')

    def translate_system_prompts(self,
                                 system_prompt_cache_path='output/system_prompt_translations.jsonl',
                                 force_retranslate=False) -> 'OrcaTranslator':
        dir_check(system_prompt_cache_path)
        if os.path.exists(system_prompt_cache_path) and not force_retranslate:
            self.logger.info(f'Found prepared system prompt translations in {system_prompt_cache_path}. Loading them.')
            with jsonlines.open(system_prompt_cache_path, 'r') as reader:
                for translation in reader:
                    self.system_prompt_translations.append(translation)
        else:
            self.logger.info(f'No prepared system prompt translations found. Collecting and translating them.')
            # Collect unique prompts
            unique_prompts = set(
                datapoint.en.system_prompt for datapoint in
                self.load_datapoints(SupportedRunPhase.SystemPromptTranslation))
            # Translate the prompts
            translate_system_prompt_gpt4 = partial(self._translate_system_prompt, model=SupportedModel.GPT4)
            with Pool(self.num_workers) as pool:
                with tqdm(total=len(unique_prompts), desc='Translating system prompts: ') as pbar:
                    for prompt_pair in pool.imap(translate_system_prompt_gpt4, unique_prompts):
                        pbar.update()
                        self.system_prompt_translations.append(prompt_pair)
            # Dump the translation pairs
            with jsonlines.open(system_prompt_cache_path, 'w') as writer:
                writer.write_all(self.system_prompt_translations)
        return self

    def _translate_system_prompt(self, system_prompt: str, model: SupportedModel) -> dict[str, str]:
        self.logger.info(f'Process {os.getpid()} is translating system prompt {system_prompt}.')
        if not system_prompt:
            return {
                'en': '',
                'zh': ''
            }
        return {
            'en': system_prompt,
            'zh': self.request_model(
                question=self.prompt_config['translate_system_prompt']['question'].format(system_prompt),
                model=model)
        }

    @retry(wait=wait_random_exponential(min=10, max=60), stop=stop_after_attempt(10), reraise=True)
    def translate_questions(self) -> 'OrcaTranslator':
        # Distribute the work to multiple processes
        self.logger.info(f'Distributing work of question translation to {self.num_workers} workers.')

        translate_question_gpt4 = partial(self._translate_question, model=SupportedModel.GPT4)

        datapoint_buffer: DataBuffer = DataBuffer(size=self.buffer_size, logger=self.logger, mode=self.mode)

        num_existing_datapoints = count_existing_datapoints(mode=self.mode, path=self.datapoint_translation_only_path)
        num_finished_datapoints = 0
        try:
            with Pool(self.num_workers) as pool:
                with tqdm(total=self.num_datapoints_to_process, desc='Translating questions: ') as pbar:
                    datapoints_vanilla = self.load_datapoints(SupportedRunPhase.QuestionTranslation)
                    for datapoint in pool.imap(translate_question_gpt4, datapoints_vanilla):
                        pbar.update()
                        num_finished_datapoints += 1
                        if int(self.num_datapoints_to_process / 100) == 0:
                            self.logger.info(f'Translating questions: '
                                             f'{num_finished_datapoints} / {self.num_datapoints_to_process} '
                                             f'datapoints finished.')
                            self.logger.info(
                                f'There are {num_existing_datapoints + num_finished_datapoints} datapoints '
                                f'in {self.datapoint_translation_only_path} now.')
                        elif num_finished_datapoints % int(self.num_datapoints_to_process / 100) == 0:
                            self.logger.info(f'Translating questions: '
                                             f'{num_finished_datapoints} / {self.num_datapoints_to_process} = '
                                             f'{num_finished_datapoints / self.num_datapoints_to_process:.0%} '
                                             f'datapoints finished.')
                            self.logger.info(
                                f'There are {num_existing_datapoints + num_finished_datapoints} datapoints '
                                f'in {self.datapoint_translation_only_path} now.')
                        datapoint_buffer.add(datapoint)
        except Exception as e:
            self.logger.warning(f'Exceptions are raised during question translation. '
                                f'Dumping the remaining {len(datapoint_buffer)} datapoints and retrying.')
            raise e
        finally:
            datapoint_buffer.dump()
        self.logger.info(f'Translation completed.')
        return self

    def _translate_question(self, datapoint: Datapoint, model: SupportedModel) -> Datapoint:
        # self.logger.info(f'Process {os.getpid()} is translating question for datapoint {datapoint.id}.')
        for pair in self.system_prompt_translations:
            if pair['en'] == datapoint.en.system_prompt:
                datapoint.zh.system_prompt = pair['zh']
                break

        # Skip if the question is about foreign languages or translation
        if not contains_only_zh_en_num_punct(datapoint.en.question) \
                or not contains_only_zh_en_num_punct(datapoint.en.response) \
                or 'translate' in datapoint.en.question.lower():
            datapoint.zh.question = '<error> <foreign_lang>'
            return datapoint

        if 'punctuat' in datapoint.en.question.lower():
            datapoint.zh.question = '<error> <lang_specific>'
            return datapoint

        # 1. Translate the question
        question = self.request_model(
            question=self.prompt_config['translate_question']['question'].format(datapoint.en.question),
            system_prompt=self.prompt_config['translate_question']['system_prompt'],
            model=model)

        # 2. Rephrase and polish the question
        # question_in = f'The following sentence is translated from English. ' \
        #               f'Please rephrase and polish it so that it sounds more natural, fluent and precise in Chinese. ' \
        #               f'Remember, do not lose any information in the source sentence!\n{question}'
        # system_prompt_in = 'You are a professional proofreader. '
        # question = self.request_model(question=question_in, system_prompt=system_prompt_in, model=model)

        # 3. Clean the translated question
        translate_pattern = re.compile(r'(?s)^.*?(<trnslt>)(.*)(</trnslt>).*?$')
        question = translate_pattern.sub(r'\2', question)
        question = question.replace('<trnslt>', '').replace('</trnslt>', '').strip()
        split_question = question.split('\n')
        if '中文' in split_question[0] or '翻译' in split_question[0] or '翻译' in question:
            question = '\n'.join(split_question[1:])
        if not contains_only_zh_en_num_punct(question):
            question = '<error> <foreign_lang>'
        if '空格' in question or '大写' in question or '小写' in question:
            question = '<error> <lang_specific>'

        datapoint.zh.question = question
        return datapoint

    @retry(wait=wait_random_exponential(min=10, max=60), stop=stop_after_attempt(10), reraise=True)
    def generate_responses(self) -> 'OrcaTranslator':
        # Distribute the work to multiple processes
        self.logger.info(f'Distributing work of response generation to {self.num_workers} workers.')

        generate_response_gpt4 = partial(self._generate_response, model=SupportedModel.GPT4)

        datapoint_buffer: DataBuffer = DataBuffer(size=self.buffer_size,
                                                  dump_path=self.datapoint_complete_path,
                                                  logger=self.logger,
                                                  mode=self.mode)

        num_existing_datapoints = count_existing_datapoints(mode=self.mode, path=self.datapoint_complete_path)
        num_finished_datapoints = 0
        try:
            with Pool(self.num_workers) as pool:
                with tqdm(total=self.num_datapoints_to_process, desc='Generating responses: ') as pbar:
                    datapoints_with_translation = self.load_datapoints(SupportedRunPhase.ResponseGeneration)
                    for datapoint in pool.imap(generate_response_gpt4, datapoints_with_translation):
                        pbar.update()
                        num_finished_datapoints += 1
                        if int(self.num_datapoints_to_process / 100) == 0:
                            self.logger.info(f'Generating responses: '
                                             f'{num_finished_datapoints} / {self.num_datapoints_to_process} '
                                             f'datapoints finished.')
                            self.logger.info(
                                f'There are {num_existing_datapoints + num_finished_datapoints} datapoints '
                                f'in {self.datapoint_translation_only_path} now.')
                        elif num_finished_datapoints % int(self.num_datapoints_to_process / 100) == 0:
                            self.logger.info(f'Generating responses: '
                                             f'{num_finished_datapoints} / {self.num_datapoints_to_process} = '
                                             f'{num_finished_datapoints / self.num_datapoints_to_process:.0%} '
                                             f'datapoints finished.')
                            self.logger.info(
                                f'There are {num_existing_datapoints + num_finished_datapoints} datapoints '
                                f'in {self.datapoint_translation_only_path} now.')
                        datapoint_buffer.add(datapoint)
        except Exception as e:
            self.logger.warning(f'Exceptions are raised during response generation. '
                                f'Dumping the remaining {len(datapoint_buffer)} datapoints and retrying.')
            raise e
        finally:
            datapoint_buffer.dump()

        self.logger.info(f'Generation completed.')

        self.logger.info('OrcaTranslator finished.')
        return self

    def _generate_response(self, datapoint: Datapoint, model: SupportedModel) -> Datapoint:
        if datapoint.zh.question.startswith('<error>'):
            return datapoint
        # self.logger.info(f'Process {os.getpid()} is generating response for datapoint {datapoint.id}.')
        response = self.request_model(question=datapoint.zh.question,
                                      system_prompt=datapoint.zh.system_prompt,
                                      model=model)
        datapoint.zh.response = response
        return datapoint

    @retry(wait=wait_random_exponential(min=10, max=60), reraise=True,
           retry=retry_if_result(retry_condition))
    def request_model(self, question: str, system_prompt=None, model=SupportedModel.GPT4) -> str:
        match model:
            case SupportedModel.GPT4:
                request_ip = "http://120.92.10.46:8080/chat"
            case SupportedModel.ChatGPT:
                request_ip = "http://47.254.22.102:8989/chat"
            case _:
                raise ValueError(f'Invalid model selected. Supported models are {list(SupportedModel)}.')

        messages = []
        if system_prompt:
            messages.append({
                "role": 'system',
                'content': system_prompt
            })
        messages.append({
            "role": 'user',
            'content': question
        })

        response = requests.post(request_ip, json={
            "model": str(model),
            "messages": messages,
            "temperature": 1,
            "top_p": 1,
            "max_tokens": 1024,
            "presence_penalty": 0,
            "frequency_penalty": 0
        })
        response_data = response.json()
        response.close()

        # print('New response: ', json.dumps(response_data, indent=4, ensure_ascii=False))
        if 'error' in response_data:
            if response_data['error'] == 'server error':
                return f'<error> <server_error>'
            elif response_data['error'] == 'busy':
                return f'<error> <busy>'
            if isinstance(response_data['error'], str):
                print(response_data['error'])
            return f"<error> <{response_data['error']['code']}> {response_data['error']['message']}"

        if response_data["choices"][0]['finish_reason'] == 'content_filter':
            return f'<error> <content_filter>'

        if 'content' not in response_data["choices"][0]["message"]:
            return f'<error> <empty_content>'

        return response_data["choices"][0]["message"]["content"].strip()
