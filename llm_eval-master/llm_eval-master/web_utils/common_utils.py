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


def show_input_df(file):
    from multi_choice import utilities
    df = utilities.load_questions(question_file=file.name)
    df = pd.DataFrame.from_records(df)
    return df.head(), gr.update(visible=True)


def gr_file_upload(upload_file):
    file_save_location = "./tmp/" # + os.path.basename(upload_file.name)
    # with open(file_save_location, "wb") as f:
    #     f.write(upload_file.read())
    # # return "上传成功"
    target_path = os.path.join(file_save_location, os.path.basename(upload_file.name))
    if os.path.exists(target_path):
        os.remove(target_path)
    shutil.copyfile(upload_file.name, target_path)


def show_slider(upload_file):
    from question_answer.common import load_model_answers

    upload_file = upload_file.name
    if upload_file.split('.')[-1] == "json" or upload_file.split('.')[-1] == 'jsonl':
        question_num = len(load_model_answers(upload_file))
        return gr.update(visible=True, value=question_num, minimum=0, maximum=question_num, step=1)
    elif upload_file.split('.')[-1] == "csv":
        question_num = len(pd.read_csv(upload_file))
        return gr.update(visible=True, value=question_num, minimum=0, maximum=question_num, step=1)
    else:
        gr.Error("文件格式解析失败。请上传jsonl或csv文件")
        return None