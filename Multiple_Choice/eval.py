import os
import sys
from utils.data_pre import Data
import re
import numpy as np
import pandas as pd
import time
from utils.data_utils import  save,postprocess
import os
from utils.plot import draw_radar_chart
import sys 


def model_init(model_name,model_path):
    if model_name in ["chatglm6b" ,"chatglm2_6b",'chatglm2_6b_finetuing']:
        from model.chatglm import Chatglm
        evaluator=Chatglm(model_path)
        return evaluator
    elif model_name=="llama_13b":
        from model.llama_13b import LLama
        evaluator = LLama(model_path)
        return evaluator
    elif model_name=="llama2_13b":
        from model.llama2_13b_hf import LLama2_13b_chat
        evaluator = LLama2_13b_chat(model_path)
        return evaluator
    elif model_name == "chatgpt":
        from model.chatgpt import Chatgpt
        evaluator = Chatgpt()
        return evaluator
    elif model_name=="qwen_7b_chat":
        from model.qianwen import Qwen
        evaluator=Qwen(model_path)
        return evaluator
    elif model_name=="internlm-chat-7b":
        from model.internlm import ILM
        evaluator=ILM(model_path)
        return evaluator
    return None

def main(ntrain,api_key,cot,temperature,model_name,model_path,cuda_visible_device):
    pro_path=os.path.dirname(__file__)
    os.environ['CUDA_VISIBLE_DEVICES']=cuda_visible_device
    if ntrain==0:
        cot=False
    data=Data(path=pro_path,ntrain=ntrain)
    p=data.preprocess()
    evaluator = model_init(model_name, model_path)
    c=0
    dic = {0: "错误", 1: "正确"}
    s_t=time.time()
    for key in p:
        result=[]
        true_false=[]
        correct_num=0
        c += 1
        check_path=os.path.join(pro_path,'result',f"{model_name}",f"ntrain={ntrain}",f"cot={cot}",f'{key}_res.csv')
        if os.path.exists(check_path):
            print(f"{check_path}已经存在，跳过！")
            continue
        for i in range(p[key]['origin'].shape[0]):
            prompt=p[key]['origin']['input'].iloc[i]
            response_str = evaluator.forward(prompt, temperature)
            print("模型输出：", response_str)
            true_ans = p[key]['origin']['output'].iloc[i]
            print("正确答案：", true_ans)
            correct_num, correct = postprocess(response_str, cot, true_ans, ntrain, correct_num)
            result.append(response_str)
            true_false.append(correct)
            print(f"正在处理学科{key}, 总进度{c}/{len(p)}, 学科进度{i + 1}/{len(p[key]['origin'])}", " ",
                    f"该题回答{dic[correct]}!, 共用时：{time.time() - s_t}s")
            print("==================================================================================")
            print()

        save(pro_path, key, result, true_false, ntrain, cot, model_name)
        print(f"       ⭐️⭐️⭐️⭐️⭐️学科{key}处理完毕，准确率{round(correct_num / len(p[key]['origin']) * 100, 2)}⭐️⭐️⭐️⭐️⭐️")
    #计算准确率
    for key in p:
        check_path = os.path.join(pro_path, 'result', f"{model_name}", f"ntrain={ntrain}", f"cot={cot}", f'{key}_res.csv')
        df=pd.read_csv(check_path)[['category','correct']]
        df=df.rename(columns={"correct":"acc"})
        map_df=df.groupby("category").mean().T 
        map_df['ALL']=map_df.mean(axis=1)['acc']
        map_df=np.round(map_df*100,2)
        map_df=map_df.T 
        filename = f'ntrain={ntrain}_cot={cot}_{model_name}'
        p1=os.path.join(pro_path, 'res_total', f"csv")
        if not os.path.exists(p1):
            os.makedirs(p1)
        path_ = os.path.join(p1, f'{filename}.csv')
        map_df.to_csv(path_)

    print(f'恭喜你，测评完成！, 总用时{time.time() - s_t}s')


if __name__ == "__main__":

    main(ntrain=5,
         api_key="",  # if model is chatgpt ,this paarameter is necessary.
         cot=False,
         temperature=0.2,
         model_name="qwen_7b_chat",
         model_path="/data2/Qwen-7b-chat",
         cuda_visible_device="0,1"
         )
