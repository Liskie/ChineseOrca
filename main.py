import os

import fire
from rich import traceback

from translator import OrcaTranslator

traceback.install(show_locals=True, width=os.get_terminal_size().columns)

if __name__ == '__main__':
    fire.Fire(OrcaTranslator)
