import time

import fire
import wandb
from collections import deque

import yaml


def count_lines(file_path):
    with open(file_path, "r") as file:
        return sum(1 for _ in file)


def calculate_moving_average_speed(speed_queue, previous_line_counts, current_line_counts, window_size: int = 10):
    speed1 = current_line_counts[0] - previous_line_counts[0]
    speed2 = current_line_counts[1] - previous_line_counts[1]

    speed_queue.append((speed1, speed2))
    if len(speed_queue) > window_size:
        speed_queue.popleft()

    avg_speed1 = sum(s[0] for s in speed_queue) / len(speed_queue)
    avg_speed2 = sum(s[1] for s in speed_queue) / len(speed_queue)

    return avg_speed1, avg_speed2


def main(log_interval: int = 10, window_size: int = 10):
    # Load path config
    with open('config/path.yaml', 'r') as reader:
        path_config = yaml.load(reader, Loader=yaml.FullLoader)

    # Load wandb API key
    with open(path_config['config_paths']['key_config_file'], 'r') as reader:
        key_config = yaml.load(reader, Loader=yaml.FullLoader)

    if not key_config['wandb_api_key']:
        raise ValueError("Please provide a valid wandb API key in config/key.yaml")

    # Initialize wandb
    wandb.login(key=key_config['wandb_api_key'])
    run = wandb.init(project="ChineseOrca")

    previous_line_counts = [0, 0]
    speed_queue = deque()

    try:
        while True:
            current_line_counts = [
                count_lines(path_config['data_paths']['datapoint_translation_only_path']),
                count_lines(path_config['data_paths']['datapoint_complete_path'])
            ]
            question_translation_speed, response_generation_speed = calculate_moving_average_speed(
                speed_queue, previous_line_counts, current_line_counts, window_size)

            # Log results to wandb
            wandb.log({"Question Translation Speed": question_translation_speed})
            wandb.log({"Response Generation Speed": response_generation_speed})

            previous_line_counts = current_line_counts

            time.sleep(log_interval)
    except KeyboardInterrupt:
        print("Terminating Speedometer.")
    finally:
        run.finish()


if __name__ == "__main__":
    fire.Fire(main)
