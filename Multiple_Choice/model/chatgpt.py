import openai
from time import sleep

class Chatgpt:
    def __init__(self):
        pass

    def forward(self,full_prompt,api_key,temperature):
        openai.api_key = api_key
        response = None
        timeout_counter = 0
        print("模型输入：",full_prompt)
        while response is None and timeout_counter <= 30:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=full_prompt,
                    temperature=temperature
                )
            except Exception as msg:
                if "timeout=600" in str(msg):
                    timeout_counter += 1
                print(msg)
                sleep(5)
                continue
        if response == None:
            response_str = ""
        else:
            response_str = response['choices'][0]['message']['content']
        return response_str