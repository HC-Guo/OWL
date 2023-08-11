"""
Usage:
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --mode [single|pairwise-baseline|pairwise-all]
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import openai
import numpy as np
from tqdm import tqdm
import os
import gradio as gr

from question_answer.common import (
    DEFAULT_MODEL_LIST,
    NEED_REF_CATS,
    Judge,
    MatchPair,
    MatchSingle,
    check_data,
    load_judge_prompts,
    load_model_answers,
    load_questions,
    play_a_match_pair,
    play_a_match_single,
)
from question_answer import configs


def make_match(
    questions,
    models,
    model_answers,
    judge,
    baseline_model,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        # 开放多轮对话的judgment，但评分的时候，仅会用全两轮的回答
        # if multi_turn and len(q["turns"]) != 2:
        #     continue
        for i in range(len(models)):
            q_id = q[configs.QUESTION_ID_COL]
            m_1 = models[i]
            m_2 = baseline_model
            if m_1 == m_2:
                continue
            a_1 = model_answers[m_1][q_id]
            a_2 = model_answers[baseline_model][q_id]
            if ref_answers is not None:
                ref = ref_answers[judge.ref_based][q_id]
                match = MatchPair(
                    dict(q),
                    m_1,
                    m_2,
                    a_1,
                    a_2,
                    judge,
                    ref_answer=ref,
                    multi_turn=multi_turn,
                )
            else:
                match = MatchPair(
                    dict(q), m_1, m_2, a_1, a_2, judge, multi_turn=multi_turn
                )
            matches.append(match)
    return matches


def make_match_all_pairs(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        # if multi_turn and len(q[configs.QUESTION_COL]) != 2:
        #     continue
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                q_id = q[configs.QUESTION_ID_COL]
                m_1 = models[i]
                m_2 = models[j]
                a_1 = model_answers[m_1][q_id]
                a_2 = model_answers[m_2][q_id]
                if ref_answers is not None:
                    ref = ref_answers[judge.ref_based][q_id]
                    match = MatchPair(
                        dict(q),
                        m_1,
                        m_2,
                        a_1,
                        a_2,
                        judge,
                        ref_answer=ref,
                        multi_turn=multi_turn,
                    )
                else:
                    match = MatchPair(
                        dict(q), m_1, m_2, a_1, a_2, judge, multi_turn=multi_turn
                    )
                matches.append(match)
    return matches


def make_match_single(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        # if multi_turn and len(q[configs.QUESTION_COL]) != 2:
        #     continue
        for i in range(len(models)):
            q_id = q[configs.QUESTION_ID_COL]
            m = models[i]
            a = model_answers[m][q_id]
            if ref_answers is not None:
                ref = ref_answers[judge.ref_based][q_id]
                matches.append(
                    MatchSingle(
                        dict(q), m, a, judge, ref_answer=ref, multi_turn=multi_turn
                    )
                )
            else:
                matches.append(MatchSingle(
                    dict(q), m, a, judge, multi_turn=multi_turn))
    return matches


def make_judge_pairwise(judge_model, judge_prompts, ref_based=None):
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["pair-v2"])
    judges["math"] = Judge(
        judge_model,
        judge_prompts["pair-math-v1"],
        ref_based=ref_based
    )
    judges["default-ref"] = Judge(
        judge_model,
        judge_prompts["pair-ref-v1"],
        ref_based=ref_based
    )

    judges["default-mt"] = Judge(
        judge_model,
        judge_prompts["pair-v2-multi-turn"],
        multi_turn=True
    )
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["pair-math-v1-multi-turn"],
        ref_based=ref_based,
        multi_turn=True,
    )
    judges["default-ref-mt"] = Judge(
        judge_model,
        judge_prompts["pair-ref-v1-multi-turn"],
        ref_based=ref_based,
        multi_turn=True,
    )
    return judges


def make_judge_single(judge_model, judge_prompts, ref_based=None):
    judges = {}
    judges["default"] = Judge(
        judge_model,
        judge_prompts["single-v1"])
    judges["math"] = Judge(
        judge_model,
        judge_prompts["single-math-v1"],
        ref_based=ref_based)
    judges["default-ref"] = Judge(
        judge_model,
        judge_prompts["single-ref-v1"],
        ref_based=ref_based
    )

    judges["default-mt"] = Judge(
        judge_model,
        judge_prompts["single-v1-multi-turn"],
        multi_turn=True
    )
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["single-math-v1-multi-turn"],
        ref_based=ref_based,
        multi_turn=True,
    )
    judges["default-ref-mt"] = Judge(
        judge_model,
        judge_prompts["single-ref-v1-multi-turn"],
        ref_based=ref_based,
        multi_turn=True,
    )
    return judges


def gen_judgment(bench_name,
                 judge_model,
                 baseline_model,
                 mode,
                 model_list,
                 reference_answer,
                 parallel,
                 start_n,
                 end_n,
                 answer_dir=None,
                 output_file=None,):

    question_file = f"./question_answer/data/{bench_name}/question.jsonl"
    answer_dir = f"./question_answer/data/{bench_name}/model_answer" if answer_dir is None else answer_dir
    ref_answer_file = f"./question_answer/data/{bench_name}/reference_answer/{reference_answer}.jsonl"

    # Load questions
    questions = load_questions(question_file, start_n, end_n)

    # Load model answers
    model_answers = {}
    for model in model_list:
        model_answers[model] = load_model_answers(
            os.path.join(answer_dir, model + ".jsonl")
        )

    # Load baseline answers
    if baseline_model is not None:
        model_answers[baseline_model] = load_model_answers(
            os.path.join(answer_dir, baseline_model + ".jsonl")
        )

    # Load reference answers
    ref_answers = {os.path.basename(ref_answer_file[:-6]): load_model_answers(
        ref_answer_file)} if reference_answer is not None else None

    # Load judge
    judge_prompts = load_judge_prompts("./question_answer/data/judge_prompts.jsonl")

    # if first_n:
    #     questions = questions[: first_n]

    models = model_list

    if mode == "single":
        make_match_func = make_match_single
        baseline_model = None

        judges = make_judge_single(judge_model, judge_prompts, reference_answer)
        play_a_match_func = play_a_match_single
        output_file = (
            f"./question_answer/data/{bench_name}/model_judgment/{judge_model}_single.jsonl"
        ) if output_file is None else output_file
    else:
        if mode == "pairwise-all":
            make_match_func = make_match_all_pairs
            baseline_model = None
        else:
            make_match_func = make_match
            baseline_model = baseline_model

        judges = make_judge_pairwise(judge_model, judge_prompts, reference_answer)
        play_a_match_func = play_a_match_pair
        output_file = (
            f"./question_answer/data/{bench_name}/model_judgment/{judge_model}_pair.jsonl"
        ) if output_file is None else output_file

    check_data(questions, model_answers, ref_answers, models, judges)

    # question_math = [q for q in questions if q["category"] in NEED_REF_CATS and len(q['question']) == 1]
    # question_math_mt = [q for q in questions if q["category"] in NEED_REF_CATS and len(q['question']) > 1]
    #
    # question_default = [q for q in questions if q["category"] not in NEED_REF_CATS and len(q['question']) == 1]
    # question_default_mt = [q for q in questions if q["category"] not in NEED_REF_CATS and len(q['question']) > 1]

    question_default = [q for q in questions if len(q['question']) == 1]
    question_default_mt = [q for q in questions if len(q['question']) > 1]

    # Make matches
    matches = []
    matches += make_match_func(
        question_default,
        models,
        model_answers,
        judges["default"] if reference_answer is None else judges["default-ref"],
        baseline_model,
        ref_answers,
    )
    matches += make_match_func(
        question_default_mt,
        models,
        model_answers,
        judges["default-mt"] if reference_answer is None else judges["default-ref-mt"],
        baseline_model,
        ref_answers,
        multi_turn=True,
    )
    # if len(question_math) and (ref_answers is None or len(ref_answers) == 0):
    #     print("math question needs reference answers.")
    # else:
    #     matches += make_match_func(
    #         question_math,
    #         models,
    #         model_answers,
    #         judges["math"],
    #         baseline_model,
    #         ref_answers,
    #     )
    # if len(question_math) and (ref_answers is None or len(ref_answers) == 0):
    #     print("math-mt question needs reference answers.")
    # else:
    #     matches += make_match_func(
    #         question_math_mt,
    #         models,
    #         model_answers,
    #         judges["math-mt"],
    #         baseline_model,
    #         ref_answers,
    #         multi_turn=True,
    #     )

    match_stat = {}
    match_stat["bench_name"] = bench_name
    match_stat["mode"] = mode
    match_stat["judge"] = judge_model
    match_stat["baseline"] = baseline_model
    match_stat["reference_answer"] = reference_answer
    match_stat["model_list"] = models
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_num_matches"] = len(matches)
    match_stat["output_path"] = output_file

    # Show match stats and prompt enter to continue
    print("Stats:")
    print(json.dumps(match_stat, indent=4))
    # input("Press Enter to confirm...")

    # Play matches
    if parallel == 1:
        # if gr_progress is None:
        print("Judging...")
        for match in tqdm(matches):
            play_a_match_func(match, output_file=output_file)
        # else:
        #     for match in gr_progress.tqdm(matches, total=len(matches), desc="Judging"):
        #         play_a_match_func(match, output_file=output_file)
    else:

        def play_a_match_wrapper(match):
            play_a_match_func(match, output_file=output_file)

        np.random.seed(0)
        np.random.shuffle(matches)

        with ThreadPoolExecutor(parallel) as executor:
            for match in tqdm(
                    executor.map(play_a_match_wrapper, matches), total=len(matches)
            ):
                pass


if __name__ == "__main__":
    import os
    # os.environ["OPENAI_API_KEY"] = "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"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="ops_bench",
        help="The name of the benchmark question set.",
    )
    # parser.add_argument(
    #     "--judge-file",
    #     type=str,
    #     default="data/judge_prompts.jsonl",
    #     help="The file of judge prompts.",
    # )
    parser.add_argument("--judge-model", type=str, default="gpt-3.5-turbo")  # gpt-3.5-turbo
    parser.add_argument("--baseline-model", type=str, default=None)  # gpt-3.5-turbo
    parser.add_argument(
        "--mode",
        type=str,
        default="pairwise-baseline",
        choices=["pairwise-baseline", "pairwise-all", "single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline` runs pairwise comparision against a baseline. "
            "`pairwise-all` runs pairwise comparision between all pairs. "
            "`single` runs single answer grading."
        ),
    )
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--reference-answer",
        type=str,
        default=None,
        # default="expert",
        help="The reference basemodel name.",
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument(
        "--start_n", type=int, help="A debug option. Only run the first `n` judgments."
    )
    parser.add_argument(
        "--end_n", type=int, help="A debug option. Only run the first `n` judgments."
    )
    parser.add_argument(
        "--openai_api_key", type=str, default=None, help="The OpenAI API key."
    )
    args = parser.parse_args()

    # # debug
    # args.bench_name = "mt_bench"
    # args.judge_file = "data/judge_prompts.jsonl"
    # args.judge_model = "gpt-3.5-turbo"
    # args.mode = "pairwise-baseline"
    # args.baseline_model = "moss-moon-003-sft"
    # args.model_list = ["chatglm2-6b"]  # , "moss-moon-003-sft"
    # args.reference_answer = None
    # args.parallel = 1
    # args.first_n = 30

    if args.openai_api_key is None:
        openai.api_key = input("Please enter your OpenAI API key: ")
    else:
        openai.api_key = args.openai_api_key

    if args.model_list is None or len(args.model_list) == 0:
        raise Exception("Please specify at least one model to be evaluated.")

    gen_judgment(args.bench_name,
                 # args.judge_file,
                 args.judge_model,
                 args.baseline_model,
                 args.mode,
                 args.model_list,
                 args.reference_answer,
                 args.parallel,
                 args.start_n,
                 args.end_n)

    # question_file = f"data/{args.bench_name}/question.jsonl"
    # answer_dir = f"data/{args.bench_name}/model_answer"
    # ref_answer_file = f"data/{args.bench_name}/reference_answer/{args.reference_answer}.jsonl"
    #
    # # Load questions
    # questions = load_questions(question_file, None, None)
    #
    #
    # # Load model answers
    # model_answers = {}
    # for model in args.model_list:
    #     model_answers[model] = load_model_answers(
    #         os.path.join(answer_dir, model + ".jsonl")
    #     )
    #
    # # Load baseline answers
    # if args.baseline_model is not None:
    #     model_answers[args.baseline_model] = load_model_answers(
    #         os.path.join(answer_dir, args.baseline_model + ".jsonl")
    #     )
    #
    # # Load reference answers
    # ref_answers = {os.path.basename(ref_answer_file[:-6]): load_model_answers(
    #     ref_answer_file)} if args.reference_answer is not None else None
    #
    # # Load judge
    # judge_prompts = load_judge_prompts(args.judge_file)
    #
    # if args.first_n:
    #     questions = questions[: args.first_n]
    #
    # models = args.model_list
    # # if args.model_list is None:
    # #     models = DEFAULT_MODEL_LIST[args.bench_name]
    # # else:
    # #     models = args.model_list
    #
    # if args.mode == "single":
    #     make_match_func = make_match_single
    #     baseline_model = None
    #
    #     judges = make_judge_single(args.judge_model, judge_prompts, args.reference_answer)
    #     play_a_match_func = play_a_match_single
    #     output_file = (
    #         f"data/{args.bench_name}/model_judgment/{args.judge_model}_single.jsonl"
    #     )
    # else:
    #     if args.mode == "pairwise-all":
    #         make_match_func = make_match_all_pairs
    #         baseline_model = None
    #     else:
    #         make_match_func = make_match
    #         baseline_model = args.baseline_model
    #
    #     judges = make_judge_pairwise(args.judge_model, judge_prompts, args.reference_answer)
    #     play_a_match_func = play_a_match_pair
    #     output_file = (
    #         f"data/{args.bench_name}/model_judgment/{args.judge_model}_pair.jsonl"
    #     )
    #
    # check_data(questions, model_answers, ref_answers, models, judges)
    #
    # question_math = [q for q in questions if q["category"] in NEED_REF_CATS and len(q['question']) == 1]
    # question_math_mt = [q for q in questions if q["category"] in NEED_REF_CATS and len(q['question']) > 1]
    #
    # question_default = [q for q in questions if q["category"] not in NEED_REF_CATS and len(q['question']) == 1]
    # question_default_mt = [q for q in questions if q["category"] not in NEED_REF_CATS and len(q['question']) > 1]
    #
    # # Make matches
    # matches = []
    # matches += make_match_func(
    #     question_default,
    #     models,
    #     model_answers,
    #     judges["default"] if args.reference_answer is None else judges["default-ref"],
    #     baseline_model,
    #     ref_answers,
    # )
    # matches += make_match_func(
    #     question_default_mt,
    #     models,
    #     model_answers,
    #     judges["default-mt"] if args.reference_answer is None else judges["default-ref-mt"],
    #     baseline_model,
    #     ref_answers,
    #     multi_turn=True,
    # )
    # if len(question_math) and (ref_answers is None or len(ref_answers) == 0):
    #     print("math question needs reference answers.")
    # else:
    #     matches += make_match_func(
    #         question_math,
    #         models,
    #         model_answers,
    #         judges["math"],
    #         baseline_model,
    #         ref_answers,
    #     )
    # if len(question_math) and (ref_answers is None or len(ref_answers) == 0):
    #     print("math-mt question needs reference answers.")
    # else:
    #     matches += make_match_func(
    #         question_math_mt,
    #         models,
    #         model_answers,
    #         judges["math-mt"],
    #         baseline_model,
    #         ref_answers,
    #         multi_turn=True,
    #     )
    #
    # match_stat = {}
    # match_stat["bench_name"] = args.bench_name
    # match_stat["mode"] = args.mode
    # match_stat["judge"] = args.judge_model
    # match_stat["baseline"] = baseline_model
    # match_stat["reference_answer"] = args.reference_answer
    # match_stat["model_list"] = models
    # match_stat["total_num_questions"] = len(questions)
    # match_stat["total_num_matches"] = len(matches)
    # match_stat["output_path"] = output_file
    #
    # # Show match stats and prompt enter to continue
    # print("Stats:")
    # print(json.dumps(match_stat, indent=4))
    # # input("Press Enter to confirm...")
    #
    # # Play matches
    # if args.parallel == 1:
    #     for match in tqdm(matches):
    #         play_a_match_func(match, output_file=output_file)
    # else:
    #
    #     def play_a_match_wrapper(match):
    #         play_a_match_func(match, output_file=output_file)
    #
    #     np.random.seed(0)
    #     np.random.shuffle(matches)
    #
    #     with ThreadPoolExecutor(args.parallel) as executor:
    #         for match in tqdm(
    #             executor.map(play_a_match_wrapper, matches), total=len(matches)
    #         ):
    #             pass
