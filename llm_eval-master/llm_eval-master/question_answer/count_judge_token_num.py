import json

import tiktoken
import pandas as pd
from tqdm import tqdm
import argparse

from question_answer.common import load_judge_prompts


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        # print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        # print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def run_judge(system_prompt, user_prompt, judgment, judge_model="gpt-3.5-turbo-0613"):
    input_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]
    output_messages = [
        {
            "role": "assistant",
            "content": judgment,
        }
    ]

    input_token_num = num_tokens_from_messages(input_messages, judge_model)
    output_token_num = num_tokens_from_messages(output_messages, judge_model)

    return input_token_num, output_token_num


def count_token_num(bench_name, model_id: list, mode, judge_model):
    if mode == "single":
        assert len(model_id) == 1
    if mode == "pair":
        assert len(model_id) == 2

    judgment_file = f'./question_answer/data/{bench_name}/model_judgment/{judge_model}_{mode}.jsonl'
    df_judgment = pd.read_json(judgment_file, lines=True)

    # Load judge
    judge_prompts = load_judge_prompts("./question_answer/data/judge_prompts.jsonl")

    detail_info = []
    if mode == "single":

        df_judgment = df_judgment[df_judgment["model"].isin(model_id)]
        for i in tqdm(range(len(df_judgment))):
            system_prompt = judge_prompts["single-v1"]["system_prompt"]
            user_prompt = df_judgment.iloc[i, :]["user_prompt"]
            judgment = df_judgment.iloc[i, :]["judgment"]

            input_token_num, output_token_num = run_judge(system_prompt, user_prompt, judgment, judge_model)

            # print(system_prompt)
            # print(user_prompt)
            # print(judgment)

            detail_info.append({"question_id": df_judgment.iloc[i, :]["question_id"],
                                "model": model_id,
                                "judge_model": judge_model,
                                "input_token_num": input_token_num,
                                "output_token_num": output_token_num,
                                })
    elif mode == "pair":
        df_judgment = df_judgment[(df_judgment["model_1"].isin(model_id)) & (df_judgment["model_2"].isin(model_id))]
        for i in tqdm(range(len(df_judgment))):
            system_prompt = judge_prompts["pair-v2"]["system_prompt"]
            user_prompt_1 = df_judgment.iloc[i, :]["g1_user_prompt"]
            judgment_1 = df_judgment.iloc[i, :]["g1_judgment"]

            input_token_num, output_token_num = run_judge(system_prompt, user_prompt_1, judgment_1, judge_model)

            # print(system_prompt)
            # print(user_prompt_1)
            # print(judgment_1)

            detail_info.append({"question_id": df_judgment.iloc[i, :]["question_id"],
                                "model": model_id,
                                "judge_model": judge_model,
                                "input_token_num": input_token_num,
                                "output_token_num": output_token_num,
                                })

            user_prompt_2 = df_judgment.iloc[i, :]["g2_user_prompt"]
            judgment_2 = df_judgment.iloc[i, :]["g2_judgment"]

            input_token_num, output_token_num = run_judge(system_prompt, user_prompt_2, judgment_2, judge_model)

            # print(system_prompt)
            # print(user_prompt_2)
            # print(judgment_2)

            detail_info.append({"question_id": df_judgment.iloc[i, :]["question_id"],
                                "model": model_id,
                                "judge_model": judge_model,
                                "input_token_num": input_token_num,
                                "output_token_num": output_token_num,
                                })

    return pd.DataFrame.from_records(detail_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench_name", type=str, default="alpaca_bench")
    parser.add_argument("--model_id", type=str, nargs='+', default=["chatglm-6b"])
    parser.add_argument("--mode", type=str, default="single")
    parser.add_argument("--judge_model", type=str, default="gpt-3.5-turbo")
    args = parser.parse_args()

    # # debug
    # args.bench_name = "alpaca_bench"
    # args.model_id = ["chatglm-6b"]
    # args.mode = "single"
    # args.judge_model = "gpt-3.5-turbo"

    df_detail = count_token_num(args.bench_name, args.model_id, args.mode, args.judge_model)
    print(f"Input tokens: {df_detail['input_token_num'].sum()}")
    print(f"Output tokens: {df_detail['output_token_num'].sum()}")
    print(df_detail)

    # total count
    info = []
    for bench in ["alpaca_bench", "ops_bench"]:
        judge_model = "gpt-3.5-turbo"
        model_list = ["chatglm-6b", "chatglm2-6b", "llama2", "moss-moon-003-sft"]

        mode = "single"
        for model_id in model_list:
            # for mode in ["single", "pairwise"]:
            print(f"bench: {bench}, model_id: {model_id}, mode: {mode}")
            df_detail = count_token_num(bench, [model_id], mode, judge_model)
            print(f"Input tokens: {df_detail['input_token_num'].sum()}")
            print(f"Output tokens: {df_detail['output_token_num'].sum()}")

            info.append({"bench": bench,
                         "model": json.dumps(model_id),
                         "mode": mode,
                         "input_token_num": df_detail['input_token_num'].sum(),
                         "output_token_num": df_detail['output_token_num'].sum()})

        mode = "pair"
        for i in range(len(model_list)):
            for j in range(i+1, len(model_list)):
                model_1 = model_list[i]
                model_2 = model_list[j]
                model_id = [model_1, model_2]
                print(f"bench: {bench}, model_id: {model_id}, mode: {mode}")
                df_detail = count_token_num(bench, model_id, mode, judge_model)
                print(f"Input tokens: {df_detail['input_token_num'].sum()}")
                print(f"Output tokens: {df_detail['output_token_num'].sum()}")

                info.append({"bench": bench,
                             "model": json.dumps(model_id),
                             "mode": mode,
                             "input_token_num": df_detail['input_token_num'].sum(),
                             "output_token_num": df_detail['output_token_num'].sum()})

    df_info = pd.DataFrame.from_records(info)
    pass

