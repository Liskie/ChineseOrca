from dataclasses import dataclass
from enum import Enum


class SupportedModel(Enum):
    GPT4 = "gpt-4"
    ChatGPT = "gpt-3.5-turbo-0613"


class SupportedDatasetType(Enum):
    Huggingface = 'huggingface'
    LismbpLocal = 'local'
    HPCLocal = 'hpc_local'


class SupportedMode(Enum):
    Restart = 'restart'
    Continue = 'continue'


class SupportedRunPhase(Enum):
    SystemPromptTranslation = 'system_prompt_translation'
    QuestionTranslation = 'question_translation'
    ResponseGeneration = 'response_generation'


class OrcaValidationError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


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
