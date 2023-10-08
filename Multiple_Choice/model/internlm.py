from transformers import AutoTokenizer, AutoModelForCausalLM
import re


class ILM():
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
        self.model.eval()
    def forward(self, prompt, temperature):
        print("模型输入：",prompt)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        # a=self.model.chat(self.tokenizer, "你好", history=None)
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



