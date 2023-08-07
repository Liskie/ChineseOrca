import json
import random

import fire
from jsonlines import jsonlines


class OrcaSampler:

    def __init__(self,
                 input_file_path: str,
                 sample_file_path: str):
        self.input_file_path = input_file_path
        self.sample_file_path = sample_file_path

    def sample(self, num: int = 100) -> None:
        # Calculate the number of lines in the file
        with jsonlines.open(self.input_file_path, 'r') as reader:
            num_lines = sum(1 for _ in reader)

        indices = sorted(random.sample(list(range(num_lines)), num))

        sampled_lines = []

        with jsonlines.open(self.input_file_path, 'r') as reader:
            for i, line in enumerate(reader):
                if i in indices:
                    sampled_lines.append(line)

        with open(self.sample_file_path, 'w') as writer:
            for line in sampled_lines:
                writer.write(f'{json.dumps(line, indent=4, ensure_ascii=False)}\n')


if __name__ == '__main__':
    fire.Fire(OrcaSampler)
