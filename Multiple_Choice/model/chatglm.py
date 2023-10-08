from transformers import AutoTokenizer, AutoModel
import re


class Chatglm():
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto").half()

    def forward(self, prompt, temperature,role=False):
        if role:
            cache_ls=[]
            for i,j in enumerate(prompt):
                if i==0:
                    continue
                else:
                    cache_ls.append(j['content'])
            if len(cache_ls)%2 !=1:
                print("数据输入部分处理错误")
                exit
            history=list(zip(cache_ls[:-1][0::2],cache_ls[:-1][1::2]))
            question=cache_ls[-1]
            print("模型输入：", f"history={history},\n,question={question}")
            response, _ = self.model.chat(self.tokenizer,
                                          question,
                                          history=history,
                                          # max_length=max_length if max_length else 2048,
                                          # top_p=0.7,
                                          do_sample=False,
                                          temperature=temperature)
            return response
        else:
            print("模型输入：",prompt)
            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
            output = self.model.generate(
                **inputs,
                # attention_mask=inputs.attention_mask,
                # do_sample=do_sample,
                temperature=temperature,
                # max_length=10000,
                max_new_tokens=10,
                do_sample=False,
                # eos_token_id=tokenizer.eos_token_id,
                # pad_token_id=tokenizer.eos_token_id
            )
            response = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return response



