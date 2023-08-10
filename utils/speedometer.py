import time

import fire
import wandb

import yaml

from utils.functions import count_existing_datapoints


def calculate_speed(current_num_lines, previous_num_lines, current_time, previous_time):
    return (current_num_lines - previous_num_lines) / (current_time - previous_time)

def main(log_interval: int = 60, window_size: int = 10):
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
    run = wandb.init(
        project="ChineseOrca",
        name='Speedometer',
        resume=True
    )

    previous_num_translation_only_datapoints = 0
    previous_time_translation_only = time.time()
    previous_num_complete_datapoints = 0
    previous_time_complete = time.time()

    try:
        while True:
            current_num_translation_only_datapoints = count_existing_datapoints(
                path_config['data_paths']['datapoint_translation_only_path'])

            current_time_translation_only = time.time()

            speed_question_translation = calculate_speed(
                current_num_translation_only_datapoints,
                previous_num_translation_only_datapoints,
                current_time_translation_only,
                previous_time_translation_only
            )

            wandb.log({
                "Question Translation Datapoints": current_num_translation_only_datapoints,
                "Question Translation Speed": speed_question_translation
            })

            current_num_complete_datapoints = count_existing_datapoints(
                (path_config['data_paths']['datapoint_complete_path']))

            current_time_complete = time.time()

            speed_response_generation = calculate_speed(
                current_num_complete_datapoints,
                previous_num_complete_datapoints,
                current_time_complete,
                previous_time_complete
            )

            wandb.log({
                "Response Generation Datapoints": current_num_complete_datapoints,
                "Response Generation Speed": speed_response_generation
            })

            previous_num_translation_only_datapoints = current_num_translation_only_datapoints
            previous_time_translation_only = current_time_translation_only
            previous_num_complete_datapoints = current_num_complete_datapoints
            previous_time_complete = current_time_complete

            time.sleep(secs=log_interval)

    except KeyboardInterrupt:
        print("Terminating Speedometer.")
    finally:
        run.finish()


if __name__ == "__main__":
    fire.Fire(main)
