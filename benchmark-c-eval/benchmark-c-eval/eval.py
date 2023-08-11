import os
from utils.data_prepross import Data
import re
import numpy as np
import pandas as pd
import time
from utils.data_utils import  save,postprocess
import os
from utils.plot import draw_radar_chart

choices = ["A", "B", "C", "D"]
def model_init(model_name,model_path):
    if model_name in ["chatglm6b_chat" ,"chatglm2_6b_chat","chatglm6b_text","chatglm2_6b_text"]:
        from model.chatglm import Chatglm
        evaluator=Chatglm(model_path)
        return evaluator
    elif model_name=="moss" :
        from model.moss import Moss
        evaluator = Moss(model_path)
        return evaluator
    elif model_name=="llama_13b":
        from model.llama_13b import LLama
        evaluator = LLama(model_path)
        return evaluator
    elif model_name=="llama2_13b_chat":
        from model.llama2_13b_hf import LLama2_13b_chat
        evaluator = LLama2_13b_chat(model_path)
        return evaluator
    elif model_name=="vicuna_33b":
        from model.vicuna import Vicuna
        evaluator = Vicuna(model_path)
        return evaluator
    elif model_name == "chatgpt":
        from model.chatgpt import Chatgpt
        evaluator = Chatgpt()
        return evaluator
    elif model_name == "lora_llama2":
        from model.lora_llama2 import Lora_LLama2
        evaluator = Lora_LLama2()
        return evaluator

    return None

def main(path,ntrain,api_key,cot,temperature,model_name,model_path,cuda_visible_device,role):
    os.environ['CUDA_VISIBLE_DEVICES']=cuda_visible_device
    evaluator=model_init(model_name,model_path)
    if ntrain==0:
        cot=False
    data=Data(path=path,ntrain=ntrain,cot=cot)
    p=data.preprocess(role=role)
    c=0
    dic = {0: "错误", 1: "正确"}
    s_t=time.time()
    for key in p:
        result=[]
        true_false=[]
        correct_num=0
        c += 1
        check_path=os.path.join(path,'result',f"{model_name}",f"ntrain={ntrain}",f"cot={cot}",f'{key}_res.csv')
        if os.path.exists(check_path):
            print(f"{check_path}已经存在，跳过！")
            continue
        if not role:
            for i in range(p[key]['origin'].shape[0]):
                prompt=p[key]['origin']['input'].iloc[i]
                response_str = evaluator.forward(prompt, temperature,role=role)
                print("模型输出：", response_str)
                true_ans = p[key]['origin']['output'].iloc[i]
                print("正确答案：",true_ans)
                correct_num, correct = postprocess(response_str, cot, true_ans, ntrain, correct_num)
                result.append(response_str)
                true_false.append(correct)
                print(f"正在处理学科{key}, 总进度{c}/{len(p)}, 学科进度{i + 1}/{len(p[key]['origin'])}", " ",
                      f"该题回答{dic[correct]}!, 共用时：{time.time() - s_t}s")
                print("==================================================================================")
                print()
        else:
            prompt_ = [{
                    "role": "system",
                    "content": f"你是一个中文人工智能助手，以下是中国关于{key}考试的单项选择题，请选出其中的正确答案。"}]
            user=[]
            for i in range(p[key]['shot'].shape[0]):
                user.append({"role": "user","content": f"{p[key]['shot']['input'].iloc[i]}"})
                user.append({"role": "assistant","content": f"{p[key]['shot']['output'].iloc[i]}"})
            prompt_shot=prompt_+user
            for j in range(p[key]['origin'].shape[0]):
                ques=[{"role": "user","content": f"{p[key]['origin']['input'].iloc[j]}"}]
                prompt=prompt_shot+ques
#模型调用----------------------------------------------------------------------------
                if model_name == "chatgpt":
                    response_str=evaluator.forward(prompt,api_key=api_key,temperature=temperature)
                # if model_name in ["chatglm6b_chat" ,"chatglm2_6b_chat"]:
                else:
                    response_str = evaluator.forward(prompt, temperature,role=role)
                print("模型输出：", response_str)
                true_ans = p[key]['origin']['output'].iloc[j]
                print("正确答案：",true_ans)
                correct_num, correct = postprocess(response_str, cot, true_ans, ntrain, correct_num)
                result.append(response_str)
                true_false.append(correct)
                print(f"正在处理学科{key}, 总进度{c}/{len(p)}, 学科进度{j + 1}/{len(p[key]['origin'])}", " ",
                      f"该题回答{dic[correct]}!, 共用时：{time.time() - s_t}s")
                print("==================================================================================")
                print()

        # total_cor[key] = [round(correct_num / len(p[key]['origin']) * 100, 2)]
        save(path, key, result, true_false, ntrain, cot, model_name)
        print(f"       ⭐️⭐️⭐️⭐️⭐️学科{key}处理完毕，准确率{round(correct_num / len(p[key]['origin']) * 100, 2)}⭐️⭐️⭐️⭐️⭐️")
    #计算准确率
    total_cor = {}
    for key in p:
        check_path = os.path.join(path, 'result', f"{model_name}", f"ntrain={ntrain}", f"cot={cot}", f'{key}_res.csv')
        df_cor=pd.read_csv(check_path).iloc[:,-1].mean()
        total_cor[key]=[np.round(df_cor*100,2)]
    filename = f'ntrain={ntrain}_cot={cot}_{model_name}'
    p1=os.path.join(path, 'res_total', f"csv")
    if not os.path.exists(p1):
        os.makedirs(p1)
    path_ = os.path.join(p1, f'{filename}.csv')
    p2=os.path.join(path, 'res_total', f"pic")
    if not os.path.exists(p2):
        os.makedirs(p2)
    draw_path = os.path.join(p2, f'{filename}.png')
    df = pd.DataFrame(data=total_cor, index=['准确率']).T
    df.to_csv(path_)
    draw_radar_chart(df['准确率'].to_list(), df.index.to_list(), filename, draw_path)
    print(f'恭喜你，测评完成！, 总用时{time.time() - s_t}s')


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--ntrain", "-k", type=int, default=5)
    # parser.add_argument("--openai_key", type=str,default="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")
    # # parser.add_argument("--model_name",type=str,default="gpt-3.5-turbo")
    # parser.add_argument("--cot",action="store_false")
    # parser.add_argument("--cuda_device", type=str)
    # args = parser.parse_args()
    '''
    model_path:  moss:"/data/models/moss-moon-003-sft"
                chatglm6b:"/data/semeron/model/chatglm/chatglm_model/chatglm-6b"
                chatglm2_6b:"/data/semeron/model/chatglm2-6b"
                llama_13b:"/data/models/llama-13B"
                llama2_13b_chat: "/data/models/llama2/Llama-2-13b-chat-hf"
                vicuna: "/data/models/vicuna-33b-v1.3""
    tempeture: 0.2
    "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"

    model_name:
        ["chatgpt",
        "moss",
        "chatglm6b_chat" ,"chatglm2_6b_chat","chatglm6b_text","chatglm2_6b_text,llama2_13b_chat"
        "llama_13b",
        "vicuna_33b"
        ]
    '''

    main(path='/data/semeron/benchmark',  # 项目地址
         ntrain=5,
         api_key="",  # if model is chatgpt ,this paarameter is necessary.
         cot=False,
         temperature=0.2,
         model_name="lora_llama2",
         model_path="/data/models/llama2/Llama-2-13b-chat-hf",
         cuda_visible_device="4,5,6,7",
         role=False)