import logging
import os
from typing import Optional

import fire
import rich.progress
import yaml
from jsonlines import jsonlines

from utils import dir_check, OrcaValidationError, Datapoint


class OrcaValidator:

    def __init__(self,
                 path_config_file: str = 'config/path.yaml',
                 datapoint_vanilla_path: Optional[str] = None,
                 datapoint_translation_only_path: Optional[str] = None,
                 datapoint_complete_path: Optional[str] = None,
                 log_path: Optional[str] = None,
                 ):

        if os.path.exists(path_config_file):
            with open(path_config_file) as reader:
                path_config = yaml.load(reader, Loader=yaml.FullLoader)

            if not datapoint_vanilla_path:
                datapoint_vanilla_path = path_config['data_paths']['datapoint_vanilla_path']
            if not datapoint_translation_only_path:
                datapoint_translation_only_path = path_config['data_paths']['datapoint_translation_only_path']
            if not datapoint_complete_path:
                datapoint_complete_path = path_config['data_paths']['datapoint_complete_path']
            if not log_path:
                log_path = path_config['project_paths']['log_path']

        self.datapoint_vanilla_path = datapoint_vanilla_path
        self.datapoint_translation_only_path = datapoint_translation_only_path
        self.datapoint_complete_path = datapoint_complete_path
        self.log_path = log_path

        dir_check(log_path)
        logging.basicConfig(filename=log_path,
                            level=logging.INFO,
                            format='[%(asctime)s] [%(levelname)s] %(message)s')
        self.logger = logging.getLogger()

        self.logger.info('OrcaValidator started.')

    def validate(self, check_translation_only: bool = True, check_complete: bool = True) -> None:
        """
        This method checks the datapoints in the translation_only and complete dump_files.
        Specifically, it checks if all datapoints in the translation_only file match those in the vanilla file and
        all datapoints in the complete file match those in the translation_only file with the ids.
        :return: None
        """
        self.logger.info('Starting validation.')

        if check_translation_only:
            self.logger.info('Checking translation_only datapoints.')
            self._validate_once(input_path=self.datapoint_vanilla_path,
                                output_path=self.datapoint_translation_only_path)

        if check_complete:
            self.logger.info('Checking complete datapoints.')
            self._validate_once(input_path=self.datapoint_translation_only_path,
                                output_path=self.datapoint_complete_path)

        self.logger.info('Validation finished.')

    def _validate_once(self, input_path: str, output_path: str) -> None:
        """
        This method checks if all datapoints in the output_path match those in the input_path.
        :param input_path: The path to the input file.
        :param output_path: The path to the output file.
        :return: None
        """

        with jsonlines.open(output_path, 'r') as output_reader:
            num_datapoints = sum(1 for _ in output_reader)

        with jsonlines.open(input_path, 'r') as input_reader, jsonlines.open(output_path, 'r') as output_reader:
            with open(input_path.replace('.jsonl', '_ids.txt'), 'w') as input_id_writer, \
                    open(output_path.replace('.jsonl', '_ids.txt'), 'w') as output_id_writer:
                for i, output_datapoint_json in rich.progress.track(enumerate(output_reader),
                                                                    description=f'Validating {output_path}:',
                                                                    total=num_datapoints):
                    try:
                        input_datapoint: Datapoint = Datapoint.from_json(input_reader.read())
                        output_datapoint: Datapoint = Datapoint.from_json(output_datapoint_json)
                        input_id_writer.write(f'{input_datapoint.id}\n')
                        output_id_writer.write(f'{output_datapoint.id}\n')
                        if input_datapoint.id != output_datapoint.id:
                            self.logger.error(f'In line {i}, input {input_path} has {input_datapoint.id}, '
                                              f'but output {output_path} has {output_datapoint.id}.')
                    except StopIteration:
                        error_info = f'Output {output_path} has {num_datapoints}, which outnumbers input {input_path}.'
                        self.logger.error(error_info)
                        raise OrcaValidationError(error_info)


if __name__ == '__main__':
    fire.Fire(OrcaValidator)
