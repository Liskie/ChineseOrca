# ChineseOrca


## Introduction
    
Just a simple tool to translate the OpenOrca dataset from English to Chinese via GPT-4.

## Quick Start

You may run the dataset building process with one single command as follows:

```shell
python main.py translate_system_prompts - translate_questions - generate_responses
```

## Detailed Usage

The translation process is divided into 3 steps:

1. Translate the system prompts
2. Translate the questions
3. Generate the responses

Since the original OpenOrca dataset is very large (it contains ~1M datapoints) thus could cost a huge amount of time, you may want to run these steps separately. In this case, you may run the following commands:

Step 1, translate the system prompts. During this step, the whole original dataset will be converted into datapoints (our custom data structure which facilitates later processing steps) and the translated system prompts will be saved to a new file. Since there are only 17 different system prompts, this step will cost a relatively short time.

```shell
python main.py translate_system_prompts
```

Step 2, translate the questions. During this step, the datapoints built in the previous step will be processed and the translated questions will be saved to a new file. Since there are ~1M questions, this step will cost a relatively long time.

```shell
python main.py translate_questions
```

Step 3, generate the responses. During this step, the datapoints built in the previous step will be processed and the responses will be generated and saved to a new file. Since there are ~1M questions, this step will also cost a relatively long time.

```shell
python main.py generate_responses
```

If you wish to customize the process, here are all the parameters you are able to change: 

```shell
python main.py \
  --dataset_type="local" \
  --num_datapoints_to_process=10000 \
  --num_workers=40 \
  --buffer_size=10 \
  --mode="continue" \
  --log_path="logs/orca_translator.log" \
  translate_system_prompts - translate_questions - generate_responses
```