# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
import numpy as np
import fire
from utils.data_prepross import Data
from model.llama import Llama
import os
import time
import os
from utils.data_prepross import Data
import re
import pandas as pd
import time
from utils.data_utils import save, postprocess
from utils.plot import draw_radar_chart



def main(
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 512,
        max_batch_size: int = 4,
        max_gen_len: Optional[int] = None,
        ntrain: int = 0,
        path: str = "",
        model_name: str = "",
        cot: bool = False,
        role:bool=True
):
    os.environ['CUDA_VISIBLE_DEVICES'] ="4,5,6,7"
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    data=Data(path=path,ntrain=ntrain, cot=cot)
    p=data.preprocess(role=role)
    c=1
    dic = {0: "错误", 1: "正确"}
    s_t=time.time()
    for key in p:
        result = []
        true_false = []
        correct_num = 0
        check_path=os.path.join(path,'result',f"{model_name}",f"ntrain={ntrain}",f"cot={cot}",f'{key}_res.csv')
        if os.path.exists(check_path):
            print(f"{check_path}已经存在，跳过！")
            continue
        if not role:
            for i in range(p[key]['origin'].shape[0]):
                prompt = p[key]['origin']['input'].iloc[i]
                print("模型输入:",prompt)
                # max_gen_len = 1 if not cot else 300
                results = generator.text_completion(
                    [prompt],  # type: ignore
                    max_gen_len=500,
                    temperature=temperature,
                    top_p=top_p,
                )
                response_str = results[0]['generation']
                print("模型输出:", response_str)
                true_ans = p[key]['origin']['output'].iloc[i]
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
            user = []
            for i in range(p[key]['shot'].shape[0]):
                user.append({"role": "user", "content": f"{p[key]['shot']['input'].iloc[i]}"})
                user.append({"role": "assistant", "content": f"{p[key]['shot']['output'].iloc[i]}"})
            prompt_shot = prompt_ + user
            for j in range(p[key]['origin'].shape[0]):
                ques = [{"role": "user", "content": f"{p[key]['origin']['input'].iloc[j]}"}]
                prompt = prompt_shot + ques
                # max_gen_len = 1 if not cot else 300
                print("模型输入:",prompt)
                results = generator.chat_completion(
                    [prompt],  # type: ignore
                    max_gen_len=500,
                    temperature=temperature,
                    top_p=top_p,
                )
                response_str = results[0]['generation']["content"]
                print("模型输出:", response_str)
                true_ans = p[key]['origin']['output'].iloc[j]
                correct_num, correct = postprocess(response_str, cot, true_ans, ntrain, correct_num)
                result.append(response_str)
                true_false.append(correct)
                print(f"正在处理学科{key}, 总进度{c}/{len(p)}, 学科进度{j + 1}/{len(p[key]['origin'])}", " ",
                      f"该题回答{dic[correct]}!, 共用时：{time.time() - s_t}s")
                print("==================================================================================")
                print()
        c += 1
        save(path, key, result, true_false, ntrain, cot, model_name)
        print(f"       ⭐️⭐️⭐️⭐️⭐️学科{key}处理完毕，准确率{round(correct_num / len(p[key]['origin']) * 100, 2)}⭐️⭐️⭐️⭐️⭐️")
    #计算准确率
    total_cor = {}
    for key in p:
        check_path = os.path.join(path, 'result', f"{model_name}", f"ntrain={ntrain}", f"cot={cot}", f'{key}_res.csv')
        df_cor=pd.read_csv(check_path).iloc[:,-1].mean()
        total_cor[key]=[np.round(df_cor*100,2)]
    filename = f'ntrain={ntrain}_cot={cot}_{model_name}'
    path_ = os.path.join(path, 'res_total', f"csv", f'{filename}.csv')
    draw_path = os.path.join(path, 'res_total', f"pic", f'{filename}.tif')
    df = pd.DataFrame(data=total_cor, index=['准确率']).T
    df.to_csv(path_)
    draw_radar_chart(df['准确率'].to_list(), df.index.to_list(), filename, draw_path)
    print(f'恭喜你，测评完成！, 总用时{time.time() - s_t}s')





if __name__ == "__main__":
    fire.Fire(main)
    # torchrun --nproc_per_node 2 eval_llama2_13B_pth.py --ckpt_dir /data/models/llama2/llama-2-13b-chat-pth/llama-2-13b-chat/ --tokenizer_path /data/models/llama2/llama-2-13b-chat-pth/tokenizer.model  --max_batch_size 1  --path "/data/semeron/c-eval-master" --model_name "llama2" --cot False --ntrain 5 --max_seq_len 5000  --role True --cuda_visible_device "4,5,6,7"