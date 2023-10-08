#  Introduction for QA Test in Owl-Bench

## Fastchat setup
```
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e .
pip install openai anthropic ray
```
or
```
pip3 install fschat
```

## To use

### step 1: Data format conversion

```
python3 data_format_conversion.py --csv-path ./data/ops_bench/Raw-QA-OwlBench-cn --jsonl-path ./data/ops_bench/question.jsonl
```
For your convenience, we have converted the data in advance. 

### step 2: Generate Answers

```
python3 gen_model_answer.py --model-path model/path --model-id id
```

Main parameters：\
`--model-path`: Local path of the model, Hugging Face format. \
`--model-id` Model ID.\
`--bench-name`：Dataset name, defaults to ops_bench. \

### step 3: Generate Judgement

Scoring of test model results using large models (gpt-4, gpt-3.5-turbo)

```
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --mode [single|pairwise-baseline|pairwise-all]
```

Main parameters：\：
`--bench-name`: Dataset name, default is ops_bench.\
`--judge-model`: Scoring models, gpt-3.5-turbo or gpt-4 are recommended.\
`--model-list`: Model to test. \
`--mode`: Testing way，options are single、pairwise-baseline、pairwise-all. \
- Single means responses of the model are scored directly by the judge model. \
- Pairwise-baseline means that the responses of the model are compared with the BASELINE model. \
- Pairwise-all means a two-by-two pairing of responses from all the models being evaluated. \


### step 4: View Scoring Results

```
python3 show_result.py --mode [single|pairwise-baseline|pairwise-all]
```
To minimize the effect of position bias, if the results of two measurements of A and B models for the same question are inconsistent or one of them is a tie, then it is regarded as a tie.
Example:
```
> python show_result.py --bench-name ops_bench --judge-model gpt-3.5-turbo --mode single
Mode: single
Input file: data/ops_bench/model_judgment/gpt-3.5-turbo_single.jsonl
                        score
model             turn       
moss-moon-003-sft 1       8.9
chatglm2-6b       1       8.7

```

```
> python show_result.py --bench-name ops_bench --judge-model gpt-3.5-turbo --mode pairwise-all
Mode: pairwise-all
Input file: data/ops_bench/model_judgment/gpt-3.5-turbo_pair.jsonl
                   win  loss  tie  win_rate  loss_rate
model                                                 
moss-moon-003-sft    2     0    8       0.2        0.0
chatglm2-6b          0     2    8       0.0        0.2

```


## Data Folder Structure

Example:
```
├── data
│   ├── judge_prompts.jsonl
│   ├── ops_bench
│   │   ├── model_answer
│   │   │   ├── chatglm2-6b.jsonl
│   │   │   └── moss-moon-003-sft.jsonl
│   │   │   └── ...
│   │   ├── model_judgment
│   │   │   ├── gpt-3.5-turbo_pair.jsonl
│   │   │   └── gpt-3.5-turbo_single.jsonl
│   │   ├── model_score
│   │   │   ├── gpt-3.5-turbo_2023-07-10_17-28-48_pairwise_label.csv
│   │   │   ├── gpt-3.5-turbo_2023-07-10_17-28-48_pairwise_summary.csv
│   │   │   ├── gpt-3.5-turbo_2023-07-10_17-29-06_single_label.csv
│   │   │   └── gpt-3.5-turbo_2023-07-10_17-29-06_single_summary.csv
│   │   ├── question.jsonl
│   │   ├── reference_answer
│   │   │   └── expert.jsonl
│   │   └── Raw-QA-OwlBench-cn.csv
```
