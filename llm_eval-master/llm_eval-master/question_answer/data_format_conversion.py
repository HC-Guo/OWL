"""Generate answers with local models.

Usage:
python3 data_format_conversion.py --csv-path ./data/ops_bench/运维领域评测数据集-v1.0.csv --jsonl-path ./data/ops_bench/question.jsonl
"""

import argparse
import json
import os

import pandas as pd
import shortuuid


def csv_to_jsonl(csv_path, jsonl_path):
    df = pd.read_csv(csv_path)

    question_file = jsonl_path
    os.makedirs(os.path.dirname(question_file), exist_ok=True)
    f_question = open(question_file, 'w', encoding='utf-8')

    answer_file = os.path.join(os.path.dirname(
        question_file), 'reference_answer/expert.jsonl')
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    f_ref_ans = open(answer_file, 'w', encoding='utf-8')

    question_id = 0
    for row in df.to_dict(orient='records'):
        # question
        # print(row["instruction"])
        # print(row["input"])
        q_content = ''
        q_content = q_content + row["instruction"] if str(row["instruction"]) != 'nan' else q_content
        q_content = q_content + row["input"] if str(row["input"]) != 'nan' else q_content

        if q_content == '':
            continue

        question_id += 1

        q_dic = {"question_id": question_id,
                 "category": "generic",
                 "ops_field": row["运维领域"],
                 "task_type": row["任务类别"],
                 "question": [q_content],
                 }
        f_question.write(json.dumps(q_dic, ensure_ascii=False)+'\n')

        # answer
        a_content = row["output"]

        a_dic = {"question_id": question_id,
                 "answer_id": shortuuid.uuid(),
                 "model_id": "expert",
                 "question": [q_content],
                 "model_answer": [a_content],
                 "tstamp": 0.0
                 }
        f_ref_ans.write(json.dumps(a_dic, ensure_ascii=False)+'\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, default="./question_answer/data/ops_bench/运维领域评测数据集-v1.0.csv")
    parser.add_argument("--jsonl-path", type=str,  default="./question_answer/data/ops_bench/question.jsonl")

    args = parser.parse_args()

    csv_to_jsonl(args.csv_path, args.jsonl_path)
