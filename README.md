# Simple Artificial Intelligence Controller Interface (SimpleAICI)

参考 [`microsoft/aici`](https://github.com/microsoft/aici)，以简洁易使用的方式实现大语言模型可控生成，能够通过额外的控制信息精确控制模型的生成格式。目前支持 `vllm`, `transformers` 库 。

## 简介

假设我们要让大语言模型生成一个表单，遵循特定的格式，并且只包含五个元素，例如：

```
请列举五个摸鱼的方法
```

通常，需要使用额外的提示词来限制生成的格式：

```
请列举五个摸鱼方法，仅返回一个列表，不要有多余的解释
```

然而，大语言模型通常难以稳定地按照提示词的要求生成特定格式的文本，并且提示词可能也会因使用的模型而异，因为每个模型都倾向于以不同的格式输出以及有不同的解释说明。虽然使用 few-shot 通常能解决这个问题，但是 few-shot 会在一定程度上影响模型的输出结果，并且也大幅增加了提示词的长度。

本项目通过在模型生成的过程中动态地插入关键的格式信息，有效地控制模型的输出格式，具体有以下功能：

1. 防止模型在开头添加解释
2. 控制表单的格式（如：1. 2. 3. 或 i. ii. iii. 或 a. b. c. ）
3. 控制模型返回的表单的条数
4. 防止模型在结尾添加解释

## 示例

使用 qwen-1.5-14B-Chat-gptq-Int4 进行演示，继续使用上面的问题，并让模型用 `a. b. c. d. e.` 分点输出结果，输入数据为:

```python
datas = [
    {
        "messages": [
            {"role": "user", "content": "请列举五个摸鱼方法"}
        ],
        "add_stop_char": ['<|aici_bos|>', '\n', '\n', '\n', '\n', '\n'],
        "fixed_content": ['a. ', 'b. ', 'c. ', 'd. ', 'e. ', '<|im_end|>']
    }
]
```

其中，`messages` 对标 OpenAI 的对话模板，填写提示词或者对话记录。  

`add_stop_char` 为匹配字符，若模型新生成的 token 中包含 `add_stop_char` 中的第0个元素 (`add_stop_char[0]`)，则将 `fixed_content` 第0个元素 (`fixed_content[0]`)中的内容填充到当前生成结果中，然后删除 `add_stop_char` 与 `fixed_content` 中的第0个元素。`<|aici_bos|>`是本项目的一个特殊标记符，用作开头便要添加内容的情况。  

在本案例中，模型的生成结果以 `a. ` 开头，然后模型继续生成，直到模型生成了包含 `\n` 的 token ，便会在生成结果中追加 `b. `，然后继续生成。当遇到最后一个 `\n` 时，在生成结果中追加 `<|im_end|>` (qwen系列的停止标识)，强迫模型结束生成。

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
a. 伪装工作：假装在进行线上会议或者处理工作邮件，但实际上是在后台浏览无关紧要的网页或社交媒体。
b. 创造虚拟任务：将一些非紧急或不重要的事情标记为优先级高的任务，以制造忙碌的假象。
c. 长时间短暂休息：频繁使用休息、喝水或上厕所等借口离开座位，降低实际工作时间。
d. 分散注意力：参与看似与工作相关的讨论或活动，实则消耗时间，例如闲聊、玩桌游或组织小型团建活动。
e. 模拟高效：使用番茄工作法或其他时间管理技巧，看似在集中工作，实际上利用间隙偷偷休息或处理私事。
output tokens: 148
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
- [ ] 跟进 transformers 以及 vllm 库的更新（如果有人用的话 0.0）
- [ ] 同步一个英文版 README


