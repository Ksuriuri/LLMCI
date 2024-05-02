[English](README_en.md) | [中文](README.md) 

# Simple Artificial Intelligence Controller Interface (SimpleAICI)

Referring to [`microsoft/aici`](https://github.com/microsoft/aici), a simple and user-friendly implementation of controllable generation of large language models, enabling precise control over the model's output format through additional control information. Currently support `vllm` and `transformers` libraries.

## Introduction

Suppose we aim for a model to generate a list, adhering to a specific format and containing only five items.

```
Top 5 ways to pretend to work
```

Typically, additional prompts are needed to limit the generated format：

```
Top 5 ways to pretend to work.
Return the result as a numbered list.
Do not add explanations, only the list.
```
However, LLM often struggle to consistently generate text in a specific format as required by the prompts, and the prompt would also vary depending on the model in use, given that each model tends to add explanations and understands instructions in different ways. Although few-shot learning can typically resolve this issue, it can affect the model's output results and also increase the length of the prompts.

This project effectively controls the output format of the model by dynamically inserting key formatting information during the model generation process. Specifically:

1. Prevent the model from adding some initial explanation
2. Control the format of list (e.g. 1. 2. 3. or i. ii. iii. or a. b. c.)
3. Limit the number of list items
4. Stop the model from adding some text after the list.

## Example

Using `Meta-Llama-3-8B-Instruct` for demonstration, continue with the previous question and instruct the model to output the results in a numbered list format using `a. b. c. d. e.` . The input data is:

```python
datas = [
    {
        "messages": [
            {"role": "user", "content": "Top 5 ways to pretend to work"}
        ],
        "add_stop_char": ['<|aici_bos|>', '\n', '\n', '\n', '\n', '\n'],
        "fixed_content": ['a. ', 'b. ', 'c. ', 'd. ', 'e. ', '<|end_of_text|>']
    }
]
```

其中，`messages` 对标 OpenAI 的对话模板，填写提示词或者对话记录。  

`add_stop_char` 为匹配字符，若模型新生成的 token 中包含 `add_stop_char` 中的第0个元素 (`add_stop_char[0]`)，则将 `fixed_content` 第0个元素 (`fixed_content[0]`)中的内容填充到当前生成结果中，然后删除 `add_stop_char` 与 `fixed_content` 中的第0个元素。`<|aici_bos|>`是本项目的一个特殊标记符，用作开头便要添加内容的情况。  

在本案例中，模型的生成结果以 `a. ` 开头，然后模型继续生成，直到模型生成了包含 `\n` 的 token ，便会在生成结果中追加 `b. `，然后继续生成。当遇到最后一个 `\n` 时，在生成结果中追加 `<|end_of_text|>` (llama3的停止标识)，强迫模型结束生成。

**完整示例代码如下:**

```python
from vllm_aici import VllmAici

# qwen-1.5 config
model_path = r'Qwen/Qwen1.5-14B-Chat-GPTQ-Int4'
lora_path = None
generation_config = {
    "stop_token_ids": [151645, 151643],
    "max_tokens": 1024,
    "top_p": 0.8,
    "top_k": 20,
    "temperature": 0.95,  # 0.95
    "repetition_penalty": 1.05,
    "use_beam_search": False,
}

model = VllmAici(model_path, model_path, generation_config, lora_path, gpu_memory_utilization=0.80)

datas = [
    {
        "messages": [
            {"role": "user", "content": "请列举五个摸鱼方法"}
        ],
        "add_stop_char": ['<|aici_bos|>', '\n', '\n', '\n', '\n', '\n'],
        "fixed_content": ['a. ', 'b. ', 'c. ', 'd. ', 'e. ', '<|im_end|>']
    }
]

outputs = model.generate(datas)

for output in zip(*outputs):
    print(output[0])
    print(f'output tokens: {output[1]}\n')
```

**模型输出：**

```
a.  **The "I'm on a Conference Call" Technique**: Pretend to be on an important conference call by putting your phone on speakerphone and having a fake conversation with someone. You can even use a voice recorder to play back a pre-recorded conversation if you're feeling extra lazy.

b.  **The "I'm in a Meeting" Method**: Claim you're in a meeting by closing your door, turning off your phone's notifications, and pretending to take notes or discuss important topics with an imaginary colleague. You can even set a timer to remind yourself when the "meeting" is over.

c.  **The "I'm Doing Research" Ruse**: Tell your boss or colleagues that you're doing research for a project, but actually spend your time browsing social media, watching cat videos, or playing online games. You can even print out some random articles or papers to make it look like you're actually working.

d.  **The "I'm Trying to Fix This Technical Issue" Excuse**: Pretend that your computer or software is malfunctioning and claim you're trying to troubleshoot the problem. You can even leave your computer screen open to a fake error message or a "loading" screen to make it look like you're actually trying to fix something.

e.  **The "I'm Taking a Break" Technique**: Claim you need to take a break to recharge and come back to your work refreshed. You can even set a timer to remind yourself when your "break" is over. Just make sure not to get too comfortable, or you might find yourself taking a longer break than intended!


output tokens: 328
```

结果生成了5条方法建议，符合预期

## 使用

### vllm

1. 参考 [`vllm-project/vllm`](https://github.com/vllm-project/vllm) 的文档 [`Installation`](https://docs.vllm.ai/en/latest/getting_started/installation.html) 安装vllm   

2. 使用 `vllm_aici.py` 中的 `VllmAici` 类即可加载 vllm 支持的任意模型，模型参数设置以及数据输入格式请参考 [示例](#示例) 。更多示例详见 `vllm_aici_demo.py`

#### Note:

- vllm 的 aici 开发是基于 `vllm==0.4.1` 版本，但理论上只要 vllm 库的 `llm.llm_engine.output_processor` 的 `_process_sequence_group_outputs` 函数没有特殊改动便能够支持 `vllm>=0.4.1` 的任意版本

- 目前不支持 `beam_search`

### transformers

1. 目前支持 `tansformers>=4.38.0` 的版本 (2024/04/30)

2. 使用 `transformers_aici.py` 中的 `TransformersAici` 类即可加载 tansformers 支持的任意模型，模型参数设置以及数据输入格式详见 `transformers_aici_demo.py`

#### Note:

- tansformers 的 aici 开发是基于 `tansformers` 的 `4.38.0` 以及 `4.40.1` 版本

- 目前仅支持 `sample` 方法

## To-Do-List

- [ ] 支持 vllm 的 Beam search case 的格式控制
- [x] 支持 transformers 的 aici（仅 sample 方法）
- [ ] 支持 transformers 除 sample 外其他方法的 aici （有生之年系列）
- [ ] 跟进 transformers 以及 vllm 库的更新（如果有人需要的话 0.0）
- [ ] 同步一个英文版 README


