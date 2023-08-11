import json

import requests


def get_response(message, uid=None, temperature=0.2):
    # 构造请求数据
    data = {
        "prompt": message,
        "uid": uid,
        "max_length": 2048,
        "top_p": 0.8,
        "temperature": temperature
    }

    # 发送POST请求
    url = "http://localhost:19325/"  # 替换为实际的应用地址
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(data))

    # 解析响应数据
    if response.status_code == 200:
        result = response.json()
        # print("Response:", result["response"])
        return result["response"]
        # 其他数据字段
    else:
        print("Request failed with status code:", response.status_code)
        

if __name__=="__main__":
    # # load json file
    # with open("/data/kicky/c-eval/ceval_moss_questions.json") as f:
    #     data = json.load(f)
    # answers = {}
    # for subject in data:
    #     answers[subject] = {}
    #     questions = data[subject]
    #     for qid in questions:
    #         meta = "请从ABCD四个选项中选出确切的答案，用字母来表示。"
    #         message = questions[qid]
    #         message = meta + message
    #         print(message)
    #         ans = get_response(message, temperature=0.2)
    #         answers[subject][qid] = ans
    # # save json file
    #         with open("/data/kicky/c-eval/ceval_moss_answers_0621.json", "w") as f:
    #             json.dump(answers, f, indent=4, ensure_ascii=False)
    message = """
你好，请问你是谁？
"""
    ans = get_response(message, temperature=0.2)
