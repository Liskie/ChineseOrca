import logging
from argparse import ArgumentParser

import fire

from translator import OrcaTranslator
from utils import SupportedMode, SupportedDatasetType, SupportedRunPhase, dir_check

# parser = ArgumentParser()
#
# parser.add_argument('-d', '--dataset_type', type=SupportedDatasetType,
#                     default=SupportedDatasetType.HPCLocal)
# parser.add_argument('-n', '--num_datapoints_to_process', type=int, default=10000)
# parser.add_argument('-w', '--num_workers', type=int, default=40)
# parser.add_argument('-b', '--buffer_size', type=int, default=100)
# parser.add_argument('-m', '--mode', type=SupportedMode, default=SupportedMode.Continue)
# parser.add_argument('-p', '--run_phases', type=list[SupportedRunPhase],
#                     default=[SupportedRunPhase.SystemPromptTranslation,
#                              SupportedRunPhase.QuestionTranslation,
#                              SupportedRunPhase.ResponseGeneration],
#                     nargs='+', choices=list(SupportedRunPhase))
# parser.add_argument('-l', '--log_path', type=str, default='output/orca_translator.log')
#
#
# def main():
#     dir_check(args.log_path)
#     logging.basicConfig(filename=args.log_path,
#                         level=logging.INFO,
#                         format='[%(asctime)s] [%(levelname)s] %(message)s')
#     logger = logging.getLogger()
#
#     translator = OrcaTranslator(dataset_type=args.dataset_type,
#                                 num_datapoints_to_process=args.num_datapoints_to_process,
#                                 num_workers=args.num_workers,
#                                 buffer_size=args.buffer_size,
#                                 mode=args.mode,
#                                 logger=logger)
#
#     if SupportedRunPhase.SystemPromptTranslation in args.run_phases:
#         translator.translate_system_prompts()
#
#     if SupportedRunPhase.QuestionTranslation in args.run_phases:
#         translator.translate_questions()
#
#     if SupportedRunPhase.ResponseGeneration in args.run_phases:
#         translator.generate_responses()


if __name__ == '__main__':
    # args = parser.parse_args()
    fire.Fire(OrcaTranslator)
