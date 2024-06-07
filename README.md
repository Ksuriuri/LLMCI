[English](README.md) | [中文](README_zh.md) 

# Large Language Model Controller Interface (LLMCI)

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
        "add_stop_char": ['<|llmci_bos|>', '\n', '\n', '\n', '\n', '\n'],
        "fixed_content": ['a. ', 'b. ', 'c. ', 'd. ', 'e. ', '<|llmci_eos|>']
    }
]
```

`messages` corresponds to OpenAI's conversation template, which is filled with chat history consisting of `system`, `user`, and `assistant` roles.

`add_stop_char` is a list of stop characters. During model generation, if `add_stop_char` is not empty and the newly generated token contains `add_stop_char[0]` after decoding, the content of `fixed_content[0]` is inserted into the current generated output, and then `add_stop_char[0]` and `fixed_content[0]` are both removed. `<| llmci_bos |>` is a special marker for situations where content needs to be added at the beginning, and the `<|llmci_eos|>` marker is used to indicate the end of the content.

In this case, since `add_stop_char[0]` is `<|llmci_bos|>`, the model's generation starts with `a. `, and then continues generating until it produces a token containing `\n`, at which point it appends `b. ` to the output and continues generating. When it encounters the last `\n`, `<|llmci_eos|>` will be automatically replaced to the `eos_token` of the model and appended to the generated text, forcing the model to stop generating.

**The complete example code is as follows:**

```python
from vllm_llmci import VllmLLMci

# llama3 config
model_path = r'meta-llama/Meta-Llama-3-8B-Instruct'
lora_path = None
generation_config = {
    "stop_token_ids": [128001, 128009],
    "max_tokens": 1024,
    "top_p": 0.6,
    "top_k": 20,
    "temperature": 0.6,
    "repetition_penalty": 1.05,
    "use_beam_search": False,
}

model = VllmLLMci(model_path, model_path, generation_config, lora_path, gpu_memory_utilization=0.80)

datas = [
    {
        "messages": [
            {"role": "user", "content": "Top 5 ways to pretend to work"}
        ],
        "add_stop_char": ['<|llmci_bos|>', '\n', '\n', '\n', '\n', '\n'],
        "fixed_content": ['a. ', 'b. ', 'c. ', 'd. ', 'e. ', '<|llmci_eos|>']
    }
]

outputs = model.generate(datas)

for output in zip(*outputs):
    print(output[0])
    print(f'output tokens: {output[1]}\n')
```

**Model outputs：**

```
a.  **The "I'm on a Conference Call" Technique**: Pretend to be on an important conference call by putting your phone on speakerphone and having a fake conversation with someone. You can even use a voice recorder to play back a pre-recorded conversation if you're feeling extra lazy.

b.  **The "I'm in a Meeting" Method**: Claim you're in a meeting by closing your door, turning off your phone's notifications, and pretending to take notes or discuss important topics with an imaginary colleague. You can even set a timer to remind yourself when the "meeting" is over.

c.  **The "I'm Doing Research" Ruse**: Tell your boss or colleagues that you're doing research for a project, but actually spend your time browsing social media, watching cat videos, or playing online games. You can even print out some random articles or papers to make it look like you're actually working.

d.  **The "I'm Trying to Fix This Technical Issue" Excuse**: Pretend that your computer or software is malfunctioning and claim you're trying to troubleshoot the problem. You can even leave your computer screen open to a fake error message or a "loading" screen to make it look like you're actually trying to fix something.

e.  **The "I'm Taking a Break" Technique**: Claim you need to take a break to recharge and come back to your work refreshed. You can even set a timer to remind yourself when your "break" is over. Just make sure not to get too comfortable, or you might find yourself taking a longer break than intended!


output tokens: 328
```

The model generated 5 suggestions, meeting the requirements.

## Usage

### vllm

1. Refer to the [`Installation`](https://docs.vllm.ai/en/latest/getting_started/installation.html) of [`vllm-project/vllm`](https://github.com/vllm-project/vllm) to install vllm

2. Any model supported by vllm can be loaded by using the `VllmLLMci` class in `vllm_llmci.py`. For model parameter settings and data input formats, please refer to [Example](#example). For more examples, please refer to `vllm_llmci_demo.py`.

#### Note:

- The vllm development is based on `vllm==0.4.1`. I just replaced the `_process_sequence_group_outputs` function of `llm.llm_engine.output_processor`

- Currently not supported for `beam_search`

### transformers

1. Currently supports `transformers>=4.38.0` (2024/04/30)

2. Any model supported by transformers can be loaded by using the `TransformersLLMci` class in `transformers_llmci.py`. For model parameter settings and data input formats, please refer to `transformers_llmci_demo.py`.

#### Note:

- The transformers development is based on `transformers==4.38.0` and `4.40.1`

- Currently, only the `sample` method is supported

## To-Do-List

- [ ] Format control of Beam search case in vllm
- [x] LLMCI in transformers (sample method only)
- [ ] LLMCI in transformers except sample (May be)
- [ ] Follow updates on the transformers and vllm（If someone needs 0.0）
- [x] `README.md` in Chinese and English


