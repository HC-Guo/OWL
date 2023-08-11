
import os
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from peft import PeftConfig, PeftModel

class Lora_LLama2():
    def __init__(self):
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
        # self.model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True,device_map='auto')
        peft_model_path = '/data/models/llama2/lora-finetuned-llama2-13b-chat/1/checkpoint-6838'
        base_model_path = '/data/models/llama2/Llama-2-13b-chat-hf'
        # config = PeftConfig.from_pretrained(peft_model_path)
        model = LlamaForCausalLM.from_pretrained(base_model_path, device_map='auto')
        self.model = PeftModel.from_pretrained(model, peft_model_path)
        self.tokenizer =LlamaTokenizer.from_pretrained(base_model_path)
        # self.model = AutoModelForCausalLM.from_pretrained(model_path,device_map='auto')
        self.model.eval()
    def forward(self,prompt,temperature,role):
        print("模型输入：", prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # outputs = self.model.generate(**inputs.cuda(), do_sample=True, temperature=temperature)
        outputs = self.model.generate(input_ids=inputs['input_ids'].cuda(),attention_mask=inputs['attention_mask'].cuda(), 
                                      max_new_tokens=1)
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response




