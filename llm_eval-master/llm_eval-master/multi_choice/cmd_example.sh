python ./multi_choice/gen_model_answer.py --model gpt2 --input_path ./multi_choice/data/truthfulqa_bench/question.jsonl --end_n 10

python ./multi_choice/gen_judgment.py --models gpt2 --metrics mc bleurt bleu rouge judge info --end_n 10

python ./multi_choice/show_result.py --models gpt2 --end_n 10