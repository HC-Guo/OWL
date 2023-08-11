import numpy as np
import gradio as gr
import pandas as pd
from gradio import components
from PIL import Image
import os
from datetime import datetime
import argparse
import openai
from tqdm import tqdm
import shutil

from web_utils.common_utils import show_input_df, gr_file_upload, show_slider


def sc_eva(benchname, metrics, model_answer, is_cot):
    from single_choice.gen_judgment import gen_judgment as sc_gen_judgment
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    model_answer = pd.read_csv(model_answer.name)
    df_detail, acc = sc_gen_judgment(model_answer, is_cot)
    detail_file_path = os.path.join(f'./single_choice/data/{benchname}/model_judgment', f"{current_time}.csv")
    df_detail.to_csv(detail_file_path)

    acc = pd.DataFrame({"bench": [benchname], "acc": [acc]})
    return detail_file_path, acc, gr.update(visible=True)


def bulid_single_choice():
    sc_benchname_value = ["mmlu", "ceval"]
    sc_benchname = components.Dropdown(choices=sc_benchname_value, label="选择对应评测数据集", value="ceval")

    sc_metric_value = ["acc"]
    sc_metrics = components.CheckboxGroup(choices=sc_metric_value, label="评估指标", value=sc_metric_value)

    sc_is_cot_value = ["True", "False"]
    sc_is_cot = components.Radio(choices=sc_is_cot_value, label="是否使用cot", value="False")

    sc_file_output = gr.File(label="上传模型回答结果")

    sc_eva_button = gr.Button("开始评测")
    sc_eva_output_summary = gr.Dataframe(visible=True, interactive=False)
    sc_eva_output_detail_file = gr.File(visible=False, interactive=False, label="评测结果详情文件")
    sc_eva_button.click(sc_eva,
                        inputs=[sc_benchname, sc_metrics, sc_file_output, sc_is_cot],
                        outputs=[sc_eva_output_detail_file, sc_eva_output_summary, sc_eva_output_detail_file])

    sc_file_demo = gr.File('./single_choice/data/ceval/model_answer/chatglm2_6b.jsonl', label="单选题-模型回答结果-格式示例")

    with gr.Accordion("历史榜单"):
        pass

    with gr.Accordion("图示"):
        pass

    with gr.Accordion("数据集"):
        gr.File("./single_choice/data/ceval/question.jsonl", label="c-eval")
        gr.File("./single_choice/data/mmlu/question_fake.jsonl", label="mmlu")