import argparse
import openai
import warnings
import pandas as pd
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

from multi_choice import models
from multi_choice import utilities
from multi_choice.configs import ENGINE_MAP, OPENAI_MODELS, MODEL_ID_COL, STANDARD_COLUMN


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,  default='gpt2')
    parser.add_argument('--preset', type=str, default='qa')
    parser.add_argument('--input_path', type=str, default='./multi_choice/data/truthfulqa_bench/question.jsonl')
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--gptj_path', type=str)
    parser.add_argument('--begin_n', type=int)
    parser.add_argument('--end_n', type=int)
    args = parser.parse_args()

    questions = utilities.load_questions(question_file=args.input_path, begin=args.begin_n, end=args.end_n)
    questions = pd.DataFrame.from_records(questions)

    # standard format
    for column in set(STANDARD_COLUMN) - set(questions.columns):
        questions[column] = None
    questions[MODEL_ID_COL] = args.model

    # set torch device
    if args.device >= 0:
        torch_device = "cuda:{0}".format(args.device)
    else:
        torch_device = "cpu"

    # check for GPT-3 models and set API key
    if args.model in OPENAI_MODELS:  # OpenAI API access required
        openai.api_key = input("Enter your OpenAI API key: ")

    # populate frame with model answers
    model = args.model
    print("Running {0}!".format(model))
    output_path = os.path.join('./multi_choice/data/truthfulqa_bench/model_answer', f"{model}.jsonl")

    # gpt-3
    if model in ['ada', 'babbage', 'curie', 'davinci']:  # gpt-3 models
        models.run_GPT3(questions, model, args.preset)
        models.run_probs_GPT3(questions, model, preset=args.preset)
    # gpt-j
    elif model == 'gptj':
        if not args.gptj_path:  # model checkpoint missing
            warnings.warn("No GPT-J model checkpoint, skipping!", stacklevel=2)
        models.run_GPTJ(questions,
                        args.preset,
                        device=torch_device,
                        cache_dir=args.cache_dir,
                        check_path=args.gptj_path)
        models.run_probs_GPTJ(questions,
                              model,
                              args.preset,
                              device=torch_device,
                              cache_dir=args.cache_dir,
                              check_path=args.gptj_path)
    # gpt-neo or gpt-2 or unifiedqa
    else:
        model_instance = AutoModelForCausalLM.from_pretrained(ENGINE_MAP.get(model, model), return_dict_in_generate=True, cache_dir=args.cache_dir).to(torch_device)
        model_instance.eval()
        tokenizer = AutoTokenizer.from_pretrained(ENGINE_MAP.get(model, model), cache_dir=args.cache_dir)

        models.run_answers(questions,
                           ENGINE_MAP.get(model, model),
                           args.preset,
                           model=model_instance,
                           tokenizer=tokenizer,
                           device=torch_device,
                           cache_dir=args.cache_dir)
        models.run_probs(questions,
                         ENGINE_MAP.get(model, model),
                         model=model_instance,
                         tokenizer=tokenizer,
                         preset=args.preset,
                         device=torch_device,
                         cache_dir=args.cache_dir)

    utilities.save_questions(questions, output_path)


if __name__ == '__main__':
    main()
