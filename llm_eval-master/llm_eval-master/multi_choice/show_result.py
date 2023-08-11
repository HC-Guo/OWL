import argparse
import numpy as np
import pandas as pd
import os

from multi_choice import utilities
from multi_choice.configs import MODEL_ID_COL


def show_result(models, begin=None, end=None, iid=None):
    METRICS = ['MC1', 'MC2', 'bleu acc', 'rouge1 acc', 'BLEURT acc', 'GPT-judge acc', 'GPT-info acc']

    # for task in ['multiple_choice', 'question_answer']:
    summary = None
    for model_key in models:
        if iid is None:
            path = os.path.join(f'./multi_choice/data/truthfulqa_bench/model_judgment', f'{model_key}.jsonl')
        else:
            path = os.path.join(f'./multi_choice/data/truthfulqa_bench/model_judgment', f'{model_key}_{iid}.jsonl')
        if not os.path.exists(path):
            continue

        questions = utilities.load_questions(question_file=path, begin=begin, end=end)
        questions = pd.DataFrame.from_records(questions)

        results = questions.loc[:, [True if i in METRICS + [MODEL_ID_COL] else False for i in questions.columns]]
        for m in METRICS:
            if m not in results.columns:
                results[m] = np.nan

        if summary is None:
            summary = results
        else:
            summary = pd.concat([summary, results], ignore_index=True)

    summary = summary.groupby([MODEL_ID_COL]).mean().reset_index()

    return summary


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['gpt2'])
    parser.add_argument('--begin_n', type=int)
    parser.add_argument('--end_n', type=int)
    args = parser.parse_args()

    summary = show_result(args.models, begin=args.begin_n, end=args.end_n)
    utilities.df_to_md(summary, os.path.join(f'./multi_choice/data/truthfulqa_bench/model_score', 'summary.md'))


if __name__ == '__main__':
    main()
