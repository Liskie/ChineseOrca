import os


def dir_check(directory: str) -> None:
    # Check if the output directory exists
    if not os.path.exists(os.path.dirname(directory)):
        os.makedirs(os.path.dirname(directory))


def retry_condition(response) -> bool:
    if response.startswith('<error> <429>') or response.startswith('<error> <server_error>'):
        return True
