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

from web_utils.common_utils import show_input_df, show_slider


def mc_eva(metrics, model_answer_file, openai_key, progress=gr.Progress(track_tqdm=True)):
    from multi_choice.gen_judgment import gen_judgment as sc_gen_judgment
    from multi_choice.show_result import show_result
    from multi_choice import utilities
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if len(metrics) == 0:
        gr.Warning("请选择评估指标")
        return None, None
    if model_answer_file is None:
        gr.Warning("请上传模型回答结果")
        return None, None

    if openai_key is None or openai_key == "" or openai_key == "请输入openai key":
        gr.Error("请输入openai key")
        return None, None
    else:
        openai.api_key = openai_key

    try:
        model_answer_file = model_answer_file.name
        model_key = os.path.splitext(os.path.basename(model_answer_file))[0]

        model_judgment = sc_gen_judgment(metrics, model_key, model_answer_file=model_answer_file, begin=None, end=None, )
        output_path = os.path.join('./multi_choice/data/truthfulqa_bench/model_judgment', f"{model_key}_{current_time}.jsonl")
        utilities.save_questions(model_judgment, output_path)

        summary = show_result([model_key])

        return summary, gr.update(visible=True)
    except Exception as e:
        gr.Error(e)
        return None, None


def bulid_multi_choice():
    mc_benchname_value = ["truthfulqa_bench"]
    mc_benchname = components.Dropdown(choices=mc_benchname_value, label="选择对应评测数据集", value="truthfulqa_bench")

    mc_metric_value = ['mc', 'bleu', 'rouge', 'bleurt', 'judge', 'info']
    mc_metrics = components.CheckboxGroup(
        choices=mc_metric_value,
        value=mc_metric_value,
        label='评估指标')

    mc_openai_key = gr.Textbox(label="openai key", value="请输入openai key")

    mc_model_output_file = gr.File(label="上传模型回答结果")
    mc_model_output_df = gr.Dataframe(visible=True, interactive=False)
    mc_question_num = gr.Slider(1, 10, 2, visible=False, interactive=True, label="选择评测题目数量")
    mc_model_output_file.change(show_input_df, inputs=mc_model_output_file, outputs=[mc_model_output_df, mc_model_output_df]).\
        then(show_slider, inputs=mc_model_output_file, outputs=mc_question_num)

    mc_eva_button = gr.Button("开始评测")
    mc_eva_output = gr.Dataframe(interactive=False, label="评估结果")
    mc_eva_button.click(mc_eva, inputs=[mc_metrics, mc_model_output_file, mc_openai_key], outputs=[mc_eva_output, mc_eva_output])

    mc_file_demo = gr.File('./multi_choice/data/truthfulqa_bench/model_answer/gpt2.jsonl', label="多选题-模型回答结果-格式示例")

    with gr.Accordion("历史榜单"):
        pass

    with gr.Accordion("图示"):
        pass

    with gr.Accordion("数据集"):
        gr.File("./multi_choice/data/truthfulqa_bench/question.jsonl", label="truthfulqa_bench")

