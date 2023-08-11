import os

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
from peft import PeftConfig, PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class Vicuna():
    def __init__(self,model_path, lora_path="/data/kicky/vicuna/checkpoints/lora_epoch5_vicuna33b_0614data_0720train"):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
        self.model = PeftModel.from_pretrained(self.model, lora_path, device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model.eval()


    def forward(self,prompt,temperature,role):
        meta_instruction = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. "
        query = f"{meta_instruction}USER: {prompt} ASSISTANT: "
        inputs = self.tokenizer(query, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, do_sample=True, max_new_tokens=256, temperature=temperature)
            response = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
        return response

        # outputs = self.model.generate(**inputs, do_sample=True, temperature=temperature, top_p=0.8, repetition_penalty=1.02,
        #                          max_new_tokens=256)
        # response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        # return response


if __name__=='__main__':
    lora_path = "/data/kicky/vicuna/checkpoints/lora_epoch5_vicuna33b_0614data_0720train"
    model_path = "/data/models/vicuna-33b-v1.3"

    model = Vicuna(model_path, lora_path)

    ans = model.forward("who are you?", temperature=0.5)