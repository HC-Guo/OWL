from transformers import LlamaForCausalLM, LlamaTokenizer
import os
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


class LLama():
    def __init__(self,model_path):
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
        # self.model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True,device_map='auto')
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained( model_path, torch_dtype=torch.float16, device_map='auto')

    def forward(self,prompt,temperature,chat):
        # max_new_tokens = 1 if not cot else 500
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # outputs = self.model.generate(**inputs.cuda(), do_sample=True, temperature=temperature)
        outputs = self.model.generate(inputs['input_ids'].cuda(),attention_mask=inputs['attention_mask'].cuda(), do_sample=True, temperature=temperature,
                                      max_new_tokens=500)
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response




