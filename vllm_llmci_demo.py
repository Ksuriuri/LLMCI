from vllm_llmci import VllmLLMci
import time

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

# # llama3 config
# model_path = r'meta-llama/Meta-Llama-3-8B-Instruct'
# lora_path = None
# generation_config = {
#     "stop_token_ids": [128001, 128009],
#     "max_tokens": 1024,
#     "top_p": 0.6,
#     "top_k": 20,
#     "temperature": 0.6,
#     "repetition_penalty": 1.05,
#     "use_beam_search": False,
# }

model = VllmLLMci(model_path, model_path, generation_config, lora_path, gpu_memory_utilization=0.90)

# Test data
datas = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "请列举五个摸鱼方法，仅返回一个列表，不要有多余的解释"}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "请列举五个摸鱼方法"}
        ],
        "add_stop_char": ['<|llmci_bos|>', '\n', '\n', '\n', '\n', '\n'],
        "fixed_content": ['1. ', '2. ', '3. ', '4. ', '5. ', '<|im_end|>']
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "请列举五个摸鱼方法"}
        ],
        "add_stop_char": ['<|llmci_bos|>'],
        "fixed_content": ['这里是']
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "请列举五个摸鱼方法"}
        ],
        "add_stop_char": ['<|llmci_bos|>', '\n', '\n', '\n', '\n'],
        "fixed_content": ['a. ', 'b. ', 'c. ', '4. ', '5. ']
    },
]

# # Speed test data
# batch_size = 1
# datas = [
#     {"messages": [{"role": "user", "content": "请写一遍5000字的综述，介绍开源的搜索引擎项目"}]} for _ in range(batch_size)
# ]

start_time = time.perf_counter()
outputs = model.generate(datas)
total_token = 0
for output in zip(*outputs):
    print('--------------------------------------------------------------------------------------------------')
    print(output[0])
    print(f'output tokens: {output[1]}\n')
    total_token += output[1]
print(f'total tokens: {total_token}, {total_token / (time.perf_counter()-start_time)} token/s')
print('batch use time', time.perf_counter()-start_time)


########################### Speed test result ###########################
# GPU: 4090, model: Qwen1.5-14B-Chat-GPTQ-Int4, gpu_memory_utilization: 0.65
# batch_size = 1: 91 token/s
# batch_size = 2: 166 token/s
# batch_size = 4: 306 token/s
# batch_size = 8: 491 token/s
# batch_size = 16: 643 token/s
# batch_size = 32: 656 token/s

# GPU: 4090, model: Qwen1.5-14B-Chat-GPTQ-Int4, gpu_memory_utilization: 0.90
# batch_size = 1: 90.12 token/s
# batch_size = 2: 161.95 token/s
# batch_size = 4: 286.08 token/s
# batch_size = 8: 489.08 token/s
# batch_size = 16: 731.20 token/s
# batch_size = 32: 771.21 token/s