import re
import pandas as pd


def gen_judgment(model_answers: pd.DataFrame, is_cot:bool):
    is_correct = []
    for i in range(len(model_answers)):
        model_answer = model_answers.iloc[i]['model_output']
        true_ans = model_answers.iloc[i]['answer']
        is_correct.append(postprocess(model_answer, is_cot, true_ans))

    model_answers['is_correct'] = is_correct
    return model_answers, sum(is_correct) / len(is_correct)


def postprocess(model_answer: str, is_cot: bool, true_ans: str):
    model_answer = model_answer.strip()
    if len(model_answer) > 0:
        ans_list = extract_ans(model_answer, is_cot)
        if len(ans_list) > 0 and (ans_list[-1].lower() == true_ans.lower()):
            correct = 1
        else:
            correct = 0
    else:
        correct = 0
    return correct


def extract_ans(response_str, cot):
    pattern = [
        r"^选([A-D])",
        r"^选项([A-D])",
        r"答案是\s*选?项?\s?([A-D])",
        r"答案为\s*选?项?\s?([A-D])",
        r"答案应为\s*选?项?\s?([A-D])",
        r"答案选\s*选?项?\s?([A-D])",
        r"答案是:\s*选?项?\s?([A-D])",
        r"答案应该是:\s*选?项?\s?([A-D])",
        r"正确的一项是\s*([A-D])",
        r"答案为:\s*选?项?\s?([A-D])",
        r"答案应为:\s*选?项?\s?([A-D])",
        r"答案:\s*选?项?\s?([A-D])",
        r"答案是：\s*选?项?\s?([A-D])",
        r"答案应该是：\s*选?项?\s?([A-D])",
        r"答案为：\s*选?项?\s?([A-D])",
        r"答案应为：\s*选?项?\s?([A-D])",
        r"答案：\s*选?项?\s?([A-D])",
        r"answer\s+is\s*([A-Z])",
        r"answer\s+is\s+\(([A-D])\)",
        r"answer\s+is\s+\（([A-D])\）",
        r"answer\s+is\s?:\s+([A-D])",
        r"answer\s+is\s?：\s+([A-D])",
        r"answer\s+is\s?：\s+（([A-D])）",
        r"answer\s+is\s?:\s+（([A-D])）",
    ]
    ans_list = []
    if not cot:
        if response_str[0] in ["A", 'B', 'C', 'D']:
            ans_list.append(response_str[0])
    else:
        if len(response_str) >= 2:
            if response_str[0] in ["A", 'B', 'C', 'D'] and response_str[1] in [".", "."]:
                ans_list.append(response_str[0])
        else:
            if response_str[0] in ["A", 'B', 'C', 'D']:
                ans_list.append(response_str[0])

    for p in pattern:
        if len(ans_list) == 0:
            ans_list = re.findall(p, response_str)
        else:
            break
    return ans_list
