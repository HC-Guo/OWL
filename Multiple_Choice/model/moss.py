import os
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

class Moss():
    def __init__(self,model_path):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
        model.tie_weights()
        self.model = load_checkpoint_and_dispatch(model, model_path, device_map="auto",
                                             no_split_module_classes=["MossBlock"],
                                             dtype=torch.float16)

    def forward(self,prompt,temperature,role):
        # max_new_tokens = 1 if not cot else 500
        query = f"<|Human|>: {prompt}<eoh>\n<|MOSS|>:"
        inputs = self.tokenizer(query, return_tensors="pt")
        outputs = self.model.generate(**inputs, do_sample=True, temperature=temperature, top_p=0.8, repetition_penalty=1.02,
                                 max_new_tokens=500,pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response




