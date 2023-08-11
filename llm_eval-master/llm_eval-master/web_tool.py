import gradio as gr

from web_utils.build_single_choice import bulid_single_choice
from web_utils.build_multi_choice import bulid_multi_choice
from web_utils.build_question_answer import bulid_question_answer


def bulid_web():
    with gr.Blocks() as demo:
        gr.Markdown("模型测评")
        with gr.Tabs():
            with gr.TabItem("问答"):  # question-answer
                bulid_question_answer()
            with gr.TabItem("单选"):  # single-choice
                bulid_single_choice()
            with gr.TabItem("多选"):  # multi-choice
                bulid_multi_choice()

        gr.Markdown("""更多介绍请参见：
        https://git.cloudwise.com/cloudwisegpt/benchmark; 
        https://yunzhihui.feishu.cn/docx/DvbNdK8d5oi7QbxLsoKcb2Eingh""")
    return demo


if __name__ == '__main__':
    bulid_web().queue(concurrency_count=5).launch(server_name='10.1.21.56', debug=True)