from jsonlines import jsonlines

from utils.data_structure import Datapoint, SupportedMode
from utils.functions import dir_check


class DataBuffer:

    def __init__(self, 
                 size=100,
                 dump_path='output/datapoints_translation_only.jsonl',
                 logger=None,
                 mode=SupportedMode.Restart):
        self.buffer: list[Datapoint] = []
        self.size = size
        self.dump_path = dump_path
        self.logger = logger
        self.mode = mode
        self.num_dumped = 0

        dir_check(dump_path)
        match self.mode:
            case SupportedMode.Restart:
                with open(dump_path, 'w') as _:
                    pass
            case SupportedMode.Continue:
                pass
            case _:
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
        self.num_dumped += len(self.buffer)

    def __len__(self):
        return len(self.buffer)
