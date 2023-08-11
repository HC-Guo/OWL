import gradio as gr
import pandas as pd
from gradio import components
from PIL import Image
import os
from datetime import datetime
import argparse
import openai

from web_utils.common_utils import gr_file_upload, show_slider
from question_answer.common import reference_answer_map


def get_total_score():
    benchs = ["alpaca_bench", "ops_bench", "mt_bench"]
    judge_models = ["gpt-3.5-turbo", "gpt-4"]
    df = None
    for bench in benchs:
        df_model = None
        for judge_model in judge_models:
            score_path = f"./question_answer/data/{bench}/model_score/{judge_model}_single_summary.csv"
            if not os.path.exists(score_path):
                continue

            df_bench = pd.read_csv(score_path)
            df_bench.rename(columns={"score": f"{bench}(score_by_{judge_model})"}, inplace=True)

            if df_model is None:
                df_model = df_bench
            else:
                df_model = pd.merge(df_model, df_bench, on="model", how="outer")

        if df_model is None:
            pass
        elif df is None:
            df = df_model
        else:
            df = pd.merge(df, df_model, on="model", how="outer")

    return df


def qa_eva_single(bench_name, judge_model, openai_key, model_answer_file, end_n, ref_answer):
    if ref_answer == "True":
        ref_answer = True
    else:
        ref_answer = False
    return qa_eva("single", bench_name, judge_model, openai_key, model_answer_file.name, end_n=end_n, ref_answer=ref_answer)


def qa_eva_pairwise(bench_name, judge_model, openai_key, model_answer_file_1, model_answer_file_2, end_n, ref_answer):
    if ref_answer == "True":
        ref_answer = True
    else:
        ref_answer = False
    return qa_eva("pairwise", bench_name, judge_model, openai_key, model_answer_file_1.name, model_answer_file_2.name, end_n=end_n, ref_answer=ref_answer)


def qa_eva(mode, bench_name, judge_model, openai_key, model_answer_file_1, model_answer_file_2=None, end_n=None, ref_answer=False, progress=gr.Progress(track_tqdm=True)):
    from question_answer.gen_judgment import gen_judgment as qa_gen_judgment
    from question_answer.show_result import display_result_single, display_result_pairwise_all

    if ref_answer:
        ref_answer = reference_answer_map[bench_name]
    else:
        ref_answer = None

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    question_label = {"alpace_bench": ["task_type"], "ops_bench": ["task_type"]}  # ops_field

    if openai_key is None or openai_key == "" or openai_key == "请输入openai key":
        gr.Error("请输入openai key")
        return None, None, None
    else:
        openai.api_key = openai_key

    if mode == "single":
        model_id = os.path.splitext(os.path.basename(model_answer_file_1))[0]
        output_file = f"./question_answer/data/{bench_name}/model_judgment/{judge_model}_single_{current_time}.jsonl"
        qa_gen_judgment(bench_name,
                     judge_model,
                     None,
                     "single",
                     [model_id],
                     reference_answer=ref_answer,
                     parallel=1,
                     start_n=None,
                     end_n=end_n,
                     answer_dir=os.path.dirname(model_answer_file_1),
                     output_file=output_file,
                     )

        args = argparse.ArgumentParser().parse_args()
        args.bench_name = bench_name
        args.judge_model = judge_model
        args.question_label = question_label.get(bench_name, [])
        args.baseline_model = None
        summary = display_result_single(args, current_time)
        return output_file, summary, gr.update(visible=True)
    elif mode == "pairwise":
        model_id_1 = os.path.splitext(os.path.basename(model_answer_file_1))[0]
        model_id_2 = os.path.splitext(os.path.basename(model_answer_file_2))[0]

        if model_id_1 == model_id_2:
            gr.Error("请选择不同的模型")
            return None, None, None

        output_file = f"./question_answer/data/{bench_name}/model_judgment/{judge_model}_pair_{current_time}.jsonl"
        qa_gen_judgment(bench_name,
                     judge_model,
                     None,
                     "pairwise-all",
                     [model_id_1, model_id_2],
                     reference_answer=ref_answer,
                     parallel=1,
                     start_n=None,
                     end_n=end_n,
                     answer_dir="./tmp/",
                     output_file=output_file,
                     # gr_progress=progress
                     )

        args = argparse.ArgumentParser().parse_args()
        args.bench_name = bench_name
        args.judge_model = judge_model
        args.question_label = question_label.get(bench_name, [])
        args.baseline_model = None
        summary = display_result_pairwise_all(args, current_time)
        return output_file, summary, gr.update(visible=True)
    else:
        gr.Error("请选择评估模式")
        return None, None, None


def bulid_question_answer():
    with gr.Tabs():
        with gr.TabItem("single score"):  # question-answer
            qa_benchname_value = ["alpaca_bench", "ops_bench", "mt_beam_bench"]
            qa_benchname = components.Dropdown(choices=qa_benchname_value, label="选择对应评测数据集", value="ops_bench")

            qa_judge_model_value = ["gpt-3.5-turbo", "gpt-4"]
            qa_judge_model = components.Dropdown(choices=qa_judge_model_value, label="选择评分模型", value="gpt-3.5-turbo")

            qa_openai_key = gr.Textbox(label="openai key", value="请输入openai key")

            qa_ref_ans_value = ["True", "False"]
            qa_ref_ans = components.Radio(choices=qa_ref_ans_value, label="是否加入参考答案", value="False")

            qa_file_output = gr.File(label="上传模型回答结果")
            qa_question_num = gr.Slider(1, 10, 2, visible=False, interactive=True, label="选择评测题目数量")
            qa_file_output.change(show_slider, inputs=qa_file_output, outputs=qa_question_num)

            qa_eva_button = gr.Button("开始评测")
            qa_eva_output = gr.Dataframe(visible=True, interactive=False)
            qa_eva_output_detail_file = gr.File(visible=False, interactive=False, label="评测结果详情文件")
            qa_eva_button.click(qa_eva_single,
                                inputs=[qa_benchname, qa_judge_model, qa_openai_key, qa_file_output, qa_question_num, qa_ref_ans],
                                outputs=[qa_eva_output_detail_file, qa_eva_output, qa_eva_output_detail_file])

        with gr.TabItem("pairwise score"):  # single-choice
            qa_benchname_value = ["alpaca_bench", "ops_bench", "mt_beam_bench"]
            qa_benchname = components.Dropdown(choices=qa_benchname_value, label="选择对应评测数据集", value="ops_bench")

            qa_judge_model_value = ["gpt-3.5-turbo", "gpt-4"]
            qa_judge_model = components.Dropdown(choices=qa_judge_model_value, label="选择对应评测数据集", value="gpt-3.5-turbo")

            qa_openai_key = gr.Textbox(label="openai key", value="请输入openai key")

            qa_ref_ans_value = ["True", "False"]
            qa_ref_ans = components.Radio(choices=qa_ref_ans_value, label="是否加入参考答案", value="False")

            qa_file_output_1 = gr.File(label="上传模型1的回答结果")
            qa_file_output_2 = gr.File(label="上传模型2的回答结果")
            qa_file_output_1.upload(gr_file_upload, inputs=[qa_file_output_1])
            qa_file_output_2.upload(gr_file_upload, inputs=[qa_file_output_2])
            qa_question_num = gr.Slider(0, 1, 1, visible=False, label="选择评测题目数量")
            qa_file_output_1.change(show_slider, inputs=qa_file_output_1, outputs=qa_question_num)

            qa_eva_button = gr.Button("开始评测")
            qa_eva_output = gr.Dataframe(visible=True, interactive=False)
            qa_eva_output_detail_file = gr.File(visible=False, interactive=False, label="评测结果详情文件")
            qa_eva_button.click(qa_eva_pairwise,
                                inputs=[qa_benchname, qa_judge_model, qa_openai_key, qa_file_output_1, qa_file_output_2, qa_question_num, qa_ref_ans],
                                outputs=[qa_eva_output_detail_file, qa_eva_output, qa_eva_output_detail_file])

        qa_file_demo = gr.File('./question_answer/data/ops_bench/model_answer/llama2.jsonl',
                               label="问答题-模型回答结果-格式示例")

    with gr.Accordion("历史榜单"):
        # gr.Markdown("Look at me...")
        gr.Dataframe(get_total_score())

    with gr.Accordion("图示"):
        # gr.Markdown("Look at me...")
        benches = ["alpaca_bench", "ops_bench"]
        with gr.Row() as radar_row:
            for bench in benches:
                gr.Image(Image.open(
                    f'./question_answer/data/{bench}/model_score/gpt-3.5-turbo_single_radar.png'),
                    label=bench, show_label=True)

        with gr.Row() as heatmap_row:
            for bench in benches:
                gr.Image(Image.open(
                    f'./question_answer/data/{bench}/model_score/gpt-3.5-turbo_pairwise_each_heatmap.png'),
                    label=bench, show_label=True)

        with gr.Row() as stacked_bar_row:
            for bench in benches:
                gr.Image(Image.open(
                    f'./question_answer/data/{bench}/model_score/gpt-3.5-turbo_pairwise_each_stacked_bar.png'),
                    label=bench, show_label=True)

    with gr.Accordion("数据集"):
        gr.File('./question_answer/data/ops_bench/question.jsonl', label="ops_bench")
        gr.File('./question_answer/data/alpaca_bench/question.jsonl', label="alpaca_bench")
        gr.File('./question_answer/data/mt_bench/question.jsonl', label="mt_bench")