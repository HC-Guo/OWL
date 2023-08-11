import argparse
import openai
import warnings
import pandas as pd
import os
from tqdm import tqdm

from multi_choice import utilities
from multi_choice.configs import CORRECT_ANSWER_COL, INCORRECT_ANSWER_COL, BEST_ANSWER_COL
from multi_choice.models import set_columns, MC_calcs
from multi_choice import metrics as do_metrics


def gen_judgment(metrics, model_id, model_answer_file=None, begin=None, end=None, cache_dir=None):
    check_m = [x for x in metrics if x in ['judge', 'info']]
    if len(check_m):  # OpenAI API access required
        # openai.api_key = input("Enter your OpenAI API key: ")
        # if 'judge' in check_m:
        #     judge_name = input("Enter your finetuned GPT-judge model name: ")
        # if 'info' in check_m:
        #     info_name = input("Enter your finetuned GPT-info model name: ")

        # debug
        openai.api_key = "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"
        judge_name = "ada"
        info_name = "ada"

    # run metrics
    if model_answer_file is None:
        model_answer_file = os.path.join('./multi_choice/data/truthfulqa_bench/model_answer', f'{model_id}.jsonl')
    questions = utilities.load_questions(question_file=model_answer_file, begin=begin, end=end)
    questions = pd.DataFrame.from_records(questions)

    set_columns(questions)
    for metric in tqdm(metrics):
        questions = questions
        if metric == 'mc':  # multiple choice
            for idx in questions.index:
                scores_true = [questions.loc[idx, "model_choice_score"][num] for num in
                               questions.loc[idx, CORRECT_ANSWER_COL]]
                scores_false = [questions.loc[idx, "model_choice_score"][num] for num in
                                questions.loc[idx, INCORRECT_ANSWER_COL]]
                score_best_answer = questions.loc[idx, "model_choice_score"][questions.loc[idx, BEST_ANSWER_COL]]
                MC_calcs(questions, idx, scores_true, scores_false, score_best_answer)
        else:  # question-answer
            if metric == 'bleurt':
                questions = do_metrics.run_BLEURT(model_id, questions, cache_dir=cache_dir)
            elif metric in ['bleu', 'rouge']:
                questions = do_metrics.run_bleu_and_rouge(model_id, questions)
            elif metric in ['judge', 'info']:
                if metric == 'judge':
                    questions = do_metrics.run_end2end_GPT3(model_id, 'GPT-judge', judge_name, questions, info=False)
                else:
                    questions = do_metrics.run_end2end_GPT3(model_id, 'GPT-info', info_name, questions, info=True)
            else:
                warnings.warn("Metric {0} not known, skipping!".format(metric), stacklevel=2)
                continue

    # if questions is not None:
    #     questions['model'] = [model_id] * len(questions)
    return questions


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['neo-small'])
    parser.add_argument('--metrics', nargs='+', default=['bleu'])
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--begin_n', type=int)
    parser.add_argument('--end_n', type=int)
    args = parser.parse_args()

    for model_key in args.models:
        questions = gen_judgment(args.metrics, model_key, model_answer_file=None, begin=args.begin_n, end=args.end_n, cache_dir=args.cache_dir)
        qa_output_path = os.path.join('./multi_choice/data/truthfulqa_bench/model_judgment', f"{model_key}.jsonl")
        utilities.save_questions(questions, qa_output_path)


if __name__ == '__main__':
    main()
