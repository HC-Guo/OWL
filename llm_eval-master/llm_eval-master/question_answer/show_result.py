"""
Usage:
python3 show_result.py --mode [single|pairwise-baseline|pairwise-all|pairwise-each]
"""
import argparse

import numpy as np
import pandas as pd
import itertools
import os
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['font.family'] = 'Songti SC'  # 显示中文标签


def draw_radar_chart(variable, data:dict, title=None, save_path=None, vmin=0, vmax=1):
    # 创建数据集
    categories = variable

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles.append(angles[0])

    # 创建子图
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'polar': True})

    # 绘制雷达图的多边形
    for k, v in data.items():
        v.append(v[0])
        ax.plot(angles, v, label=k, linewidth=1, )

    # 设置刻度标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # 设置雷达图的范围
    ax.set_ylim(vmin, vmax)

    if title is not None:
        ax.set_title(title)

    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)

    # # 显示雷达图
    # plt.show()
    plt.close()


def draw_horizontal_stacked_bar_chart(categories, win, tie, loss, title=None, save_path=None):
    fig = plt.figure(figsize=(16, 12))  # 设置宽度为8英寸，高度为6英寸

    ax = fig.add_subplot(111)

    bar_win = ax.barh(categories, win, color=plt.cm.Blues(0.8), label='win')
    bar_tie = ax.barh(categories, tie, left=win, color=plt.cm.Blues(0.5), label='tie')
    bar_loss = ax.barh(categories, loss, left=np.array(win) + np.array(tie), color=plt.cm.Blues(0.2), label='loss')

    # 添加数据标签
    for bar in bar_win:
        width = bar.get_width()
        ax.text(bar.get_x() + width / 2, bar.get_y() + bar.get_height() / 2, round(width, 3), ha='center', va='center')

    for bar in bar_tie:
        width = bar.get_width()
        ax.text(bar.get_x() + width / 2, bar.get_y() + bar.get_height() / 2, round(width, 3), ha='center', va='center')

    for bar in bar_loss:
        width = bar.get_width()
        ax.text(bar.get_x() + width / 2, bar.get_y() + bar.get_height() / 2, round(width, 3), ha='center', va='center')

    ax.tick_params(axis='y', labelsize=12)

    if title is not None:
        ax.set_title(title)

    ax.legend()

    if save_path is not None:
        plt.savefig(save_path)

    # # 显示热力图
    # plt.show()
    plt.close()


def draw_heatmap(categories, data, title=None, save_path=None):
    # 设置坐标轴标签
    import seaborn as sns
    ax = sns.heatmap(data, cmap='RdYlBu', annot=True, xticklabels=categories, yticklabels=categories[::-1],
                     mask=np.isnan(data),
                     fmt='.3f', vmin=0, vmax=1)
    # 设置坐标位置为左上角
    ax.invert_yaxis()
    # 将横坐标位置调整到上方
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    # 显示特殊符号
    for text in ax.texts:
        if text.get_text() == 'nan':
            text.set_text('-')

    if title is not None:
        ax.set_title(title)

    # 保存图像到本地
    if save_path is not None:
        plt.savefig(save_path)

    # # 显示图形
    # plt.show()

    plt.close()


def df_to_md(df, path):
    # 将DataFrame转换为Markdown格式
    df_md = df.to_markdown(index=False)

    # 将Markdown格式写入.md文件
    with open(path, 'w') as f:
        f.write(df_md)


def df_to_csv(df, path):
    df.to_csv(path, index=False)


def display_result_single(args, iid=None):
    input_file = f"./question_answer/data/{args.bench_name}/model_judgment/{args.judge_model}_single.jsonl" if iid is None else \
        f"./question_answer/data/{args.bench_name}/model_judgment/{args.judge_model}_single_{iid}.jsonl"
    question_file = f"./question_answer/data/{args.bench_name}/question.jsonl"

    print(f"Input file: {input_file}")
    df_judgment = pd.read_json(input_file, lines=True)
    df = df_judgment[["question_id", "model", "score", "turn"]]

    print(f"Question file: {question_file}")
    df_question = pd.read_json(question_file, lines=True)
    question_label = [args.question_label] if isinstance(args.question_label, str) else args.question_label
    df_question = df_question[["question_id"] + question_label]

    df = df.merge(df_question, left_on="question_id", right_on="question_id")

    # 获取当前时间并生成文件名
    dir_path = f"./question_answer/data/{args.bench_name}/model_score"
    os.makedirs(dir_path, exist_ok=True)

    df_1 = df[["model", "score", ]].groupby(["model"]).mean()
    df_1.reset_index(inplace=True)
    df_1.sort_values(by="score", ascending=False, inplace=True)
    print(df_1)
    file_df_1 = os.path.join(dir_path, f"{args.judge_model}_single_summary.md") if iid is None else \
        os.path.join(dir_path, f"{args.judge_model}_single_summary_{iid}.md")
    df_to_md(df_1, file_df_1)
    file_df_1 = os.path.join(dir_path, f"{args.judge_model}_single_summary.csv") if iid is None else \
        os.path.join(dir_path, f"{args.judge_model}_single_summary_{iid}.csv")
    df_to_csv(df_1, file_df_1)

    if len(question_label) > 0:
        df_2 = df[["model", "score"] + question_label].groupby(question_label+["model"]).mean()
        df_2.reset_index(inplace=True)
        df_2.sort_values(by=question_label + ["score"], ascending=False, inplace=True)
        print(df_2)
        file_df_2 = os.path.join(dir_path, f"{args.judge_model}_single_label.md") if iid is None else \
            os.path.join(dir_path, f"{args.judge_model}_single_label_{iid}.md")
        df_to_md(df_2, file_df_2)
        file_df_2 = os.path.join(dir_path, f"{args.judge_model}_single_label.csv") if iid is None else \
            os.path.join(dir_path, f"{args.judge_model}_single_label_{iid}.csv")
        df_to_csv(df_2, file_df_2)
        return df_2

    if len(question_label) == 1:
        question_label_value = df[question_label[0]].unique()
        data = {}
        for model in df["model"].unique():
            data[model] = []
            for label in question_label_value:
                data[model].append(df[(df["model"] == model) & (df[question_label[0]] == label)]["score"].mean())
        draw_radar_chart(question_label_value,
                         data,
                         title=f"Single score | Evaluated by {args.judge_model}",
                         save_path=os.path.join(dir_path, f"{args.judge_model}_single_radar.png") if iid is None else os.path.join(dir_path, f"{args.judge_model}_single_radar_{iid}.png"),
                         vmax=10)
    return df_1


def get_result_pairwise(df_judgment, baseline_model=None):
    list_res = []
    # traverse df row by row
    for index, row in df_judgment.iterrows():
        if baseline_model is not None:
            if baseline_model not in [row["model_1"], row["model_2"]]:
                continue
        if row["g1_winner"] == "tie" or row["g1_winner"] != row["g2_winner"] or row["g1_winner"] == "error" or row["g2_winner"] == "error":
            list_res.append({"model": row["model_1"], "win": 0, "loss": 0, "tie": 1})
            list_res.append({"model": row["model_2"], "win": 0, "loss": 0, "tie": 1})
        else:
            if row["g1_winner"] == "model_1":
                winner = row["model_1"]
                loser = row["model_2"]
            else:
                winner = row["model_2"]
                loser = row["model_1"]
            list_res.append({"model": winner, "win": 1, "loss": 0, "tie": 0})
            list_res.append({"model": loser, "win": 0, "loss": 1, "tie": 0})

    df = pd.DataFrame(list_res)
    return df


def display_result_pairwise_all(args, iid=None):
    input_file = f"./question_answer/data/{args.bench_name}/model_judgment/{args.judge_model}_pair.jsonl" if iid is None else \
        f"./question_answer/data/{args.bench_name}/model_judgment/{args.judge_model}_pair_{iid}.jsonl"
    question_file = f"./question_answer/data/{args.bench_name}/question.jsonl"

    print(f"Input file: {input_file}")
    df_judgment = pd.read_json(input_file, lines=True)

    print(f"Question file: {question_file}")
    df_question = pd.read_json(question_file, lines=True)
    question_label = [args.question_label] if isinstance(args.question_label, str) else args.question_label
    df_question = df_question[["question_id"] + question_label]

    df_judgment = df_judgment.merge(df_question, left_on="question_id", right_on="question_id")

    # Scoring of total question
    df = get_result_pairwise(df_judgment, args.baseline_model)
    df = df.groupby(["model"]).sum()
    df.reset_index(inplace=True)

    # remove baseline model
    if args.baseline_model is not None:
        df = df[df["model"] != args.baseline_model]
    # add win rate
    df["win_rate"] = df["win"] / (df["win"] + df["loss"] + df["tie"])
    df["loss_rate"] = df["loss"] / (df["win"] + df["loss"] + df["tie"])
    df.sort_values(by="win_rate", ascending=False, inplace=True)
    print(df)

    dir_path = f"./question_answer/data/{args.bench_name}/model_score"
    os.makedirs(dir_path, exist_ok=True)
    file_df = os.path.join(dir_path, f"{args.judge_model}_pairwise_summary.md") if iid is None else \
        os.path.join(dir_path, f"{args.judge_model}_pairwise_summary_{iid}.md")
    df_to_md(df, file_df)
    file_df = os.path.join(dir_path, f"{args.judge_model}_pairwise_summary.csv") if iid is None else \
        os.path.join(dir_path, f"{args.judge_model}_pairwise_summary_{iid}.csv")
    df_to_csv(df, file_df)

    # Scoring of segmented areas
    if len(question_label) > 0:
        df = None
        combinations = list(itertools.product(*[df_question[label].unique().tolist() for label in question_label]))
        for comb in combinations:
            df_comb = df_judgment
            for i, label in enumerate(question_label):
                df_comb = df_comb[df_comb[label] == comb[i]]
            df_comb = get_result_pairwise(df_comb, args.baseline_model)

            if len(df_comb) == 0:
                continue

            for i, label in enumerate(question_label):
                df_comb[f"{label}"] = comb[i]

            df_comb = df_comb.groupby(question_label + ["model"]).sum()
            df_comb.reset_index(inplace=True)

            # remove baseline model
            if args.baseline_model is not None:
                df_comb = df_comb[df_comb["model"] != args.baseline_model]

            df = df_comb if df is None else pd.concat([df, df_comb])

        df["win_rate"] = df["win"] / (df["win"] + df["loss"] + df["tie"])
        df["loss_rate"] = df["loss"] / (df["win"] + df["loss"] + df["tie"])
        df.sort_values(by=question_label + ["win_rate"], ascending=False, inplace=True)
        print(df)

        file_df = os.path.join(dir_path, f"{args.judge_model}_pairwise_label.md") if iid is None else \
            os.path.join(dir_path, f"{args.judge_model}_pairwise_label_{iid}.md")
        df_to_md(df, file_df)
        file_df = os.path.join(dir_path, f"{args.judge_model}_pairwise_label.csv") if iid is None else \
            os.path.join(dir_path, f"{args.judge_model}_pairwise_label_{iid}.csv")
        df_to_csv(df, file_df)

    if len(question_label) == 1:
        question_label_value = df[question_label[0]].unique()
        data = {}
        for model in df["model"].unique():
            data[model] = []
            for label in question_label_value:
                data[model].append(df[(df["model"] == model) & (df[question_label[0]] == label)]["win_rate"].mean())
        draw_radar_chart(question_label_value,
                         data,
                         title=f"Pairwise total win rate | Evaluated by {args.judge_model}",
                         save_path=os.path.join(dir_path, f"{args.judge_model}_pairwise_total_radar.png") if iid is None else \
                             os.path.join(dir_path, f"{args.judge_model}_pairwise_total_radar_{iid}.png"),
                         vmax=1)
    return df


def display_result_pairwise_each(args):
    # 指定benchmark，生成各个模型之间的胜率。 不考虑benchmark中细分领域的胜率

    input_file = (
        f"./question_answer/data/{args.bench_name}/model_judgment/{args.judge_model}_pair.jsonl"
    )
    question_file = (
        f"./question_answer/data/{args.bench_name}/question.jsonl"
    )

    print(f"Input file: {input_file}")
    df_judgment = pd.read_json(input_file, lines=True)

    print(f"Question file: {question_file}")
    df_question = pd.read_json(question_file, lines=True)
    question_label = [args.question_label] if isinstance(args.question_label, str) else args.question_label
    df_question = df_question[["question_id"] + question_label]

    df_judgment = df_judgment.merge(df_question, left_on="question_id", right_on="question_id")

    model_list = df_judgment["model_1"].unique().tolist() + df_judgment["model_2"].unique().tolist()
    model_list = list(set(model_list))
    model_list = sorted(model_list)

    # draw horizontal stacked bar chart
    categories, win, tie, loss = [], [], [], []
    for column, baseline_model in enumerate(model_list):
        df = get_result_pairwise(df_judgment, baseline_model=baseline_model)
        df = df.groupby(["model"]).sum()
        df.reset_index(inplace=True)

        # 计算胜率，因为主角替换为baseline model，所以win和loss需要交换
        df["win_rate"] = df["loss"] / (df["win"] + df["loss"] + df["tie"])
        df["tie_rate"] = df["tie"] / (df["win"] + df["loss"] + df["tie"])
        df["loss_rate"] = df["win"] / (df["win"] + df["loss"] + df["tie"])

        df = df[df["model"] != baseline_model]

        categories.append([f"{baseline_model}\nvs\n{model}" for model in df["model"].values])
        win.append(df["win_rate"].values.tolist())
        tie.append(df["tie_rate"].values.tolist())
        loss.append(df["loss_rate"].values.tolist())

    sorted_index = np.argsort([np.mean(x) for x in win])
    categories = np.array(categories)[sorted_index].flatten()
    win = np.array(win)[sorted_index].flatten()
    tie = np.array(tie)[sorted_index].flatten()
    loss = np.array(loss)[sorted_index].flatten()

    draw_horizontal_stacked_bar_chart(categories,
                                      win,
                                      tie,
                                      loss,
                                      title=f"Pairwise score | Evaluated by {args.judge_model}",
                                      save_path=os.path.join(f"./question_answerdata/{args.bench_name}/model_score", f"{args.judge_model}_pairwise_each_stacked_bar.png"))

    model_list = [model_list[x] for x in sorted_index[::-1]]
    # draw headmap
    pairwise_win_rate = np.zeros((len(model_list), len(model_list)))
    np.fill_diagonal(pairwise_win_rate, np.nan)
    pairwise_win_rate = pairwise_win_rate[::-1, :]

    for column, baseline_model in enumerate(model_list):
        df = get_result_pairwise(df_judgment, baseline_model=baseline_model)
        df = df.groupby(["model"]).sum()
        df.reset_index(inplace=True)

        # add win rate
        df["win_rate"] = df["win"] / (df["win"] + df["loss"] + df["tie"])

        for i, model in enumerate(model_list):
            if model != baseline_model:
                pairwise_win_rate[len(model_list)-1-i, column] = df[df["model"] == model]["win_rate"].values[0]

    draw_heatmap(model_list,
                 pairwise_win_rate,
                 title=f"Pairwise score | Evaluated by {args.judge_model}",
                 save_path=os.path.join(f"./question_answer/data/{args.bench_name}/model_score", f"{args.judge_model}_pairwise_each_heatmap.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="alpaca_bench")
    # parser.add_argument("--input-file", type=str)
    parser.add_argument("--judge-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--mode",
        type=str,
        default="pairwise-all",
        choices=["pairwise-baseline", "pairwise-all", "pairwise-each",  "single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline`: Show the results of Pairwise comparison between each model and baseline models. "
            "`pairwise-all`: Show the results of Pairwise comparison between model and all other models. "
            "`pairwise-eacn`: Show the results of Pairwise comparison between model and each other models. "
            "`single`: runs single answer grading."
        ),
    )
    parser.add_argument("--question-label", type=str, default=[], nargs="+",)
    parser.add_argument("--iid", type=str, default=None, help="Specify the iid of the question to display the result.")
    args = parser.parse_args()

    # debug
    # args.bench_name = "ops_bench"
    # args.mode = "pairwise-each"
    # args.judge_model = "gpt-3.5-turbo"
    # args.question_label = ["ops_field"]

    if args.mode == "single":
        display_result_func = display_result_single
    elif args.mode == "pairwise-baseline":
        display_result_func = display_result_pairwise_all
    elif args.mode == "pairwise-all":
        args.baseline_model = None
        display_result_func = display_result_pairwise_all
    else:
        display_result_func = display_result_pairwise_each

    print(f"Mode: {args.mode}")
    display_result_func(args, iid=args.iid)
