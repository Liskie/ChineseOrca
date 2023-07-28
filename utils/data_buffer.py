from jsonlines import jsonlines

from .data_structure import Datapoint
from .utils import dir_check


class DataBuffer:

    def __init__(self, size=100, dump_path='output/datapoints_translation_only.jsonl', logger=None):
        self.buffer: list[Datapoint] = []
        self.size = size
        self.dump_path = dump_path
        self.logger = logger

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
        if not self.buffer:
            return
        self.logger.info(f'Dumping {len(self.buffer)} datapoints into {self.dump_path}.')
        with jsonlines.open(self.dump_path, 'a') as writer:
            writer.write_all([datapoint.to_json() for datapoint in self.buffer])
