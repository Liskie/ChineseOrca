import json
import random

import fire
from jsonlines import jsonlines


class OrcaSampler:

    def __init__(self,
                 input_file_path: str,
                 sample_file_path: str):
        """
        This class samples a given number of lines from a given file.
        :param input_file_path: This file should be in .jsonl format.
        :param sample_file_path: This file will be in .json format with indentation=4.
        """
        self.input_file_path = input_file_path
        self.sample_file_path = sample_file_path

    def sample(self, num: int = 100, start: int = 0, end: int = -1) -> None:
        # Calculate the number of lines in the file
        with jsonlines.open(self.input_file_path, 'r') as reader:
            num_lines = sum(1 for _ in reader)

        if end == -1:
            end = num_lines

        indices = sorted(random.sample(list(range(num_lines))[start:end], num))

        sampled_lines = []

        with jsonlines.open(self.input_file_path, 'r') as reader:
            for i, line in enumerate(reader):
                if i in indices:
                    sampled_lines.append(line)

        with open(self.sample_file_path, 'w') as writer:
            json.dump(sampled_lines, writer, indent=4, ensure_ascii=False)

        print(f'Dumped {num} lines to {self.sample_file_path}.')


if __name__ == '__main__':
    fire.Fire(OrcaSampler)
