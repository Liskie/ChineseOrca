import fire
from jsonlines import jsonlines
from rich.progress import track


class DataCleaner:

    def __init__(self, input_path: str, output_path: str = None):
        self.input_path = input_path

        if not output_path:
            self.output_path = input_path.replace('.jsonl', '_cleaned.jsonl')
        else:
            self.output_path = output_path

    def clean(self):
        total = 0
        valid = 0

        with (jsonlines.open(self.input_path, 'r') as reader):
            with jsonlines.open(self.output_path, 'w') as writer:
                for datapoint in track(reader, description='Cleaning datapoints: '):
                    total += 1

                    if datapoint['zh']['question'].startswith('<error>') or \
                            datapoint['zh']['response'].startswith('<error>'):
                        continue

                    valid += 1
                    writer.write(datapoint)

        print(f'Cleaning completed. {valid} / {total} = {valid / total:.2f}% datapoints are valid.')


if __name__ == '__main__':
    fire.Fire(DataCleaner)
