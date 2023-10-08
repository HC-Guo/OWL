
import os
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


class LLama2_13b_chat():
    def __init__(self,model_path):
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
        # self.model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True,device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path,device_map='auto')

    def forward(self,prompt,temperature,role):
        print("模型输入：", prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # outputs = self.model.generate(**inputs.cuda(), do_sample=True, temperature=temperature)
        outputs = self.model.generate(inputs['input_ids'].cuda(),attention_mask=inputs['attention_mask'].cuda(), do_sample=False, temperature=temperature,
                                      max_new_tokens=500)
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response




