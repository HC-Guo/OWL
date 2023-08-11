# LLM Judge

本自动化评估脚本主要基于[MT-Bench](https://arxiv.org/abs/2306.05685) 数据集提供的测试脚本(位于[Fastchat](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) )进行修改，用于进行大模型在运维问答数据上做自动化评估。


## Fastchat 安装
```
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e .
pip install openai anthropic ray
```
或者
```
pip3 install fschat
```

## 使用

### step 1: 数据格式转化
需要将我们的'运维领域评测数据集-v1.0'的数据格式转化为MT-Bench的数据格式，具体转化脚本位于`data_format_conbersion.py`，使用方法如下：
```
python3 data_format_conversion.py --csv-path ./data/ops_bench/运维领域评测数据集-v1.0.csv --jsonl-path ./data/ops_bench/question.jsonl
```

运维领域评测数据集-v1.0 格式如下：
```
| 运维领域     | 任务类别    | instruction     | input         | output     |
|------------|------------|-----------------|    -------    |     -------|
| 信息安全     | 知识问答    |                 |如何保护系统安全？| "以下是保护系统安全的几个关键步骤：\n1. 安装安全更新：确保系统的操作系统、应用程序和防病毒软件等全部安全更新都已安装，以确保系统不会因已知漏洞而受到攻击。\n2. 使用强密码：使用强密码并定期更改密码以确保账户安全。\n3. 防病毒和反恶意软件：安装并定期更新防病毒和反恶意软件，以确保系统不会感染恶意软件。\n4. 及时备份：定期备份重要数据以防止数据丢失和恶意软件攻击。\n5. 安装防火墙：安装并配置防火墙以防止未经授权的访问。\n6. 限制访问权限：限制用户和管理员访问权限，确保只有授权的人员才能访问系统和数据。\n7. 培训员工：对员工进行安全培训以提高他们的安全意识和防范能力。\n8. 监控日志：定期监控系统日志以及时发现异常行为和攻击。" |
```

MT-Bench 格式如下：
```
{"question_id": 100, "category": "roleplay", "turns": ["Picture yourself as a 100-years-old tree in a lush forest, minding your own business, when suddenly, a bunch of deforesters shows up to chop you down. How do you feel when those guys start hacking away at you?", "Come up with a proposal to convince the deforesters to stop cutting you down and other trees."]}
{"question_id": 101, "category": "reasoning", "turns": ["Imagine you are participating in a race with a group of people. If you have just overtaken the second person, what's your current position? Where is the person you just overtook?", "If the \"second person\" is changed to \"last person\" in the above question, what would the answer be?"], "reference": ["You are in second place.", "Uncertain."]}				
```


### step 2: 生成测试模型答案
模型答案生成脚本位于`gen_model_answer.py`，使用方法如下：
```
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
```
主要参数如下：\
`--model-path`为模型的本地路径，模型为Hugging Face的格式。\
`--model-id`为模型ID。\
`--bench-name`：为数据集名称，可选项为mt_bench和ops_bench，默认为ops_bench。\
更多参数详见代码。


### step 3: 生成测试测评结果
使用大模型（gpt-4、gpt-3.5-turbo）对测试模型的结果进行打分
```
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --mode [single|pairwise-baseline|pairwise-all]
```
主要参数如下：
`--bench-name`为数据集名称，可选项为mt_bench和ops_bench，默认为ops_bench。\
`--judge-model`为打分的大模型，推荐选用gpt-3.5-turbo或gpt-4。\
`--model-list`为被测评的模型列表。\
`--mode`为测评方式，可选项为single、pairwise-baseline、pairwise-all，默认为single。\
     single表示使用测试模型直接对被测评模型的回答进行打分，评分结果为分数值。 single模式下`--baseline-model`参数无效。\
     pairwise-baseline表示将baseline模型的回答与被测评模型的回答进行配对，让评测模型判断哪个回答更好，评分结果为哪个模型更好，非评分值。\
     pairwise-all表示将所有被测评的模型的回答进行两两配对，让评测模型判断哪个回答更好，评分结果为哪个模型更好，非评分值。pairwise-all模式下模式下`--baseline-model`参数无效。\
`--baseline-model` 使用pairwise评分方式时，会将baseline模型的回答与被测评模型的回答进行配对。\
更多参数详见代码。

使用pairwise评测方式对比模型A和B时，为降低position biase的影响，会执行两次评测：第一次A回答在前、B回答在后；第二次B回答在前、A回答在后。

### step 4: 查看评分结果
```
python3 show_result.py --mode [single|pairwise-baseline|pairwise-all]
```
测评方式为pairwise-baseline|pairwise-all时，为降低position biase的影响，若A、B模型针对同一个问题的两次测评结果不一致或者其中一次为平手，则视为平手。
示例如下：
```
> python show_result.py --bench-name ops_bench --judge-model gpt-3.5-turbo --mode single
Mode: single
Input file: data/ops_bench/model_judgment/gpt-3.5-turbo_single.jsonl
                        score
model             turn       
moss-moon-003-sft 1       8.9
chatglm2-6b       1       8.7

```

```
> python show_result.py --bench-name ops_bench --judge-model gpt-3.5-turbo --mode pairwise-all
Mode: pairwise-all
Input file: data/ops_bench/model_judgment/gpt-3.5-turbo_pair.jsonl
                   win  loss  tie  win_rate  loss_rate
model                                                 
moss-moon-003-sft    2     0    8       0.2        0.0
chatglm2-6b          0     2    8       0.0        0.2

```

show_result.py的输出结果会保存在data/{bench_name}/model_score中。

参数含义详见代码。

### step 5: (可选) 客户端对比评分结果
在网页上查看各问题下各模型的对比结果。服务启动方式如下：
```
python3 qa_browser.py
```

## data文件夹说明
当前data文件中已包含mt_bench和ops_bench两个数据集的转化结果，目录结构如下。

question.jsonl为转化后的数据集\
reference_answer为参考答案\
model_answer为模型答案\
model_judgment为模型评估结果。
```
├── data
│   ├── judge_prompts.jsonl
│   ├── mt_bench
│   │   ├── model_answer
│   │   │   ├── chatglm2-6b.jsonl
│   │   │   └── moss-moon-003-sft.jsonl
│   │   ├── model_judgment
│   │   │   └── gpt-3.5-turbo_single.jsonl
│   │   ├── model_score
│   │   │   ├── gpt-3.5-turbo_2023-07-10_17-31-33_pairwise_label.csv
│   │   │   ├── gpt-3.5-turbo_2023-07-10_17-31-33_pairwise_summary.csv
│   │   │   ├── gpt-3.5-turbo_2023-07-10_17-31-45_single_label.csv
│   │   │   └── gpt-3.5-turbo_2023-07-10_17-31-45_single_summary.csv
│   │   ├── question.jsonl
│   │   └── reference_answer
│   │       └── gpt-4.jsonl
│   ├── ops_bench
│   │   ├── model_answer
│   │   │   ├── chatglm2-6b.jsonl
│   │   │   └── moss-moon-003-sft.jsonl
│   │   ├── model_judgment
│   │   │   ├── gpt-3.5-turbo_pair.jsonl
│   │   │   └── gpt-3.5-turbo_single.jsonl
│   │   ├── model_score
│   │   │   ├── gpt-3.5-turbo_2023-07-10_17-28-48_pairwise_label.csv
│   │   │   ├── gpt-3.5-turbo_2023-07-10_17-28-48_pairwise_summary.csv
│   │   │   ├── gpt-3.5-turbo_2023-07-10_17-29-06_single_label.csv
│   │   │   └── gpt-3.5-turbo_2023-07-10_17-29-06_single_summary.csv
│   │   ├── question.jsonl
│   │   ├── reference_answer
│   │   │   └── expert.jsonl
│   │   └── 运维测试评测数据集-v1.0.csv
```

## 适配自定义的模型
fastchat中提供了一系列的大模型适配器（adapter），不同模型的adapter的system prompter、role形式、模型参数、prompt格式不同。如果需要适配自定义的模型，需要仿照`addition_model_adapter.py`中的添加moss的形式添加自定义的模型。

fastchat中提供的adapter列表如下：
```
# Note: the registration order matters.
# The one registered earlier has a higher matching priority.
register_model_adapter(VicunaAdapter)
register_model_adapter(T5Adapter)
register_model_adapter(KoalaAdapter)
register_model_adapter(AlpacaAdapter)
register_model_adapter(ChatGLMAdapter)
register_model_adapter(DollyV2Adapter)
register_model_adapter(OasstPythiaAdapter)
register_model_adapter(OasstLLaMAAdapter)
register_model_adapter(StableLMAdapter)
register_model_adapter(BaizeAdapter)
register_model_adapter(RwkvAdapter)
register_model_adapter(OpenBuddyAdapter)
register_model_adapter(PhoenixAdapter)
register_model_adapter(BardAdapter)
register_model_adapter(PaLM2Adapter)
register_model_adapter(ChatGPTAdapter)
register_model_adapter(ClaudeAdapter)
register_model_adapter(MPTAdapter)
register_model_adapter(BiLLaAdapter)
register_model_adapter(RedPajamaINCITEAdapter)
register_model_adapter(H2OGPTAdapter)
register_model_adapter(RobinAdapter)
register_model_adapter(SnoozyAdapter)
register_model_adapter(WizardLMAdapter)
register_model_adapter(ManticoreAdapter)
register_model_adapter(GuanacoAdapter)
register_model_adapter(CamelAdapter)
register_model_adapter(ChangGPTAdapter)
register_model_adapter(TuluAdapter)
register_model_adapter(FalconAdapter)
register_model_adapter(TigerBotAdapter)
register_model_adapter(BaichuanAdapter)

# After all adapters, try the default base adapter.
register_model_adapter(BaseModelAdapter)
```

## 备注
1. 使用gpt-3.5-turbo模型时,需要科学上网工具,目前159和170上有jade私人赞助的科学上网工具。
2. gpt-4目前没有提供api，评测脚本还不支持使用gpt-4。
3. 测试脚本支持多轮对话形式的问答题，目前打分脚本支持最多2轮对话的问答评分。