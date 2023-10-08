"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import time
import setproctitle
import shortuuid
import torch
from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import get_conversation_template, load_model
from tqdm import tqdm
from transformers.generation.utils import logger
from fastchat.conversation import conv_templates
from fastchat.model.model_adapter import model_adapters

import configs

logger.setLevel("INFO")

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

file_name = os.path.basename(__file__)
setproctitle.setproctitle(file_name[:-3])

update_fastchat(model_adapters, conv_templates)

def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_length,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
):
    questions = load_questions(question_file, question_begin, question_end)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    # use_ray = num_gpus_total > 1

    # if use_ray:
    #     get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
    #         get_model_answers
    #     ).remote
    # else:
    get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_length,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
            )
        )

    # if use_ray:
    #     ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_length,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
):
    model, tokenizer = load_model(
        model_path,
        device="cuda",
        num_gpus=num_gpus_per_model,
        max_gpu_memory=max_gpu_memory,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )

    logger.info(
        f"eos_token_id:{tokenizer.eos_token_id}, pad_token_id:{tokenizer.pad_token_id}")

    for question in tqdm(questions):
        if question[configs.CATEGORY_COL] in temperature_config:
            temperature = temperature_config[question[configs.CATEGORY_COL]]
        else:
            temperature = 0.7

        # for i in range(num_choices):
        # torch.manual_seed(i)
        # choices = []
        conv = get_conversation_template(model_id)
        turns = []
        for j in range(len(question[configs.QUESTION_COL])):
            qs = question[configs.QUESTION_COL][j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)

            prompt = conv.get_prompt()
            print(prompt, '\n')
            inputs = tokenizer([prompt], return_tensors="pt")

            if temperature < 1e-4:
                do_sample = False
            else:
                do_sample = True

            # start_time = time.time()
            output_ids = model.generate(
                inputs.input_ids.cuda(),
                attention_mask=inputs.attention_mask.cuda() if 'attention_mask' in inputs else None,
                do_sample=do_sample,
                temperature=temperature,
                max_length=max_length,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
            # print("generate", time.time() - start_time, '\n\n')

            if model.config.is_encoder_decoder:
                output_ids = output_ids[0]
            else:
                output_ids = output_ids[0][len(inputs.input_ids[0]):]
            output = tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )
            if conv.stop_str:
                output = output[: output.find(conv.stop_str)]
            output = output.strip()
            print(output)

            turns.append(output)
            conv.messages[-1][-1] = output

            # choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a", encoding='utf-8') as fout:
            ans_json = {
                configs.QUESTION_ID_COL: question[configs.QUESTION_ID_COL],
                configs.CATEGORY_COL:question[configs.CATEGORY_COL],
                configs.QUESTION_COL: question[configs.QUESTION_COL],
                # "answer_id": shortuuid.uuid(),
                configs.MODEL_ID_COL: model_id,
                configs.MODEL_ANSWER_COL: turns,
                configs.TSTIME_COL: time.time(),
            }
            print(ans_json)
            ans_json = json.dumps(ans_json, ensure_ascii=False)
            print(ans_json)
            fout.write(ans_json + "\n")
            # fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r", encoding="utf-8") as fin:
        for l in fin:
            qid = json.loads(l)[configs.QUESTION_ID_COL]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w", encoding="utf-8") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        # required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--model-id", 
                        type=str, 
                        # required=True
                        )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str,
                        help="The output answer file.")
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="The maximum number of total tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    args = parser.parse_args()


    if args.num_gpus_total > 1:
        import ray

        ray.init()

    question_file = f"./Question_Answer/data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"./Question_Answer/data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        args.model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_length,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
    )

    reorg_answer_file(answer_file)
