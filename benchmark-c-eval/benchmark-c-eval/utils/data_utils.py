import os
import re
import pandas as pd

def  save(path,subject,result,true_false,ntrain,cot,model_name):
    path_=os.path.join(path,'result',f"{model_name}",f"ntrain={ntrain}",f"cot={cot}")
    if not os.path.exists(path_):
        os.makedirs(path_)
    ori_path=os.path.join(path,f'data/val/{subject}_val.csv')
    ori_data=pd.read_csv(ori_path)
    ori_data['model_output']=result
    ori_data['correct']=true_false
    tar_path=os.path.join(path_, f'{subject}_res.csv')
    ori_data.to_csv(tar_path,index=False)




def postprocess(response_str,cot,true_ans,ntrain,correct_num):
    response_str = response_str.strip()
    if len(response_str) > 0:
        ans_list=extract_ans(response_str,cot)
        if len(ans_list) > 0 and (ans_list[-1].lower() == true_ans.lower()):
            correct_num+=1
            correct=1
        else:
            correct=0
    else:
        correct=0
    return correct_num,correct

def extract_ans( response_str,cot):
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
        if len(response_str)>=2:
            if response_str[0] in ["A", 'B', 'C', 'D'] and response_str[1] in [".", "."] :
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
