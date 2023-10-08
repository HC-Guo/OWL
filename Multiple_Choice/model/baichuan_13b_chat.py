
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
class Baichuan():
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
    def forward(self, prompt, temperature,role=True):
        print("模型输入：", prompt)
        response= self.model.chat(self.tokenizer,
                                  prompt,
                                  # max_length=max_length if max_length else 2048,
                                  # top_p=0.7,
                                #   do_sample=True,
                                #   temperature=temperature
                                  )
        return response