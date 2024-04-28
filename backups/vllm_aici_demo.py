from vllm_aici import VllmAici
import time

# qwen-1.5 config
model_path = r'/home/oneway/ssd2t/model/qwen/Qwen1___5-14B-Chat-GPTQ-Int4'
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
# model_path = r'/home/oneway/ssd2t/model/LLM-Research/Meta-Llama-3-8B-Instruct'
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

model = VllmAici(model_path, model_path, generation_config, lora_path, gpu_memory_utilization=0.80)

# Test data
datas = [
    {
        "messages": [
            {"role": "user", "content": "请列举五个摸鱼方法，仅返回一个列表，不要有多余的解释"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "请列举五个摸鱼方法"}
        ],
        "add_stop_char": ['<|aici_bos|>', '\n', '\n', '\n', '\n', '\n'],
        "fixed_content": ['1. ', '2. ', '3. ', '4. ', '5. ', '<|im_end|>']
    },
    {
        "messages": [
            {"role": "user", "content": "请列举五个摸鱼方法"}
        ],
        "add_stop_char": ['<|aici_bos|>'],
        "fixed_content": ['这里是']
    },
    {
        "messages": [
            {"role": "user", "content": "请列举五个摸鱼方法"}
        ],
        "add_stop_char": ['<|aici_bos|>', '\n', '\n'],
        "fixed_content": ['a. ', 'b. ', 'c. ']
    },
]

# # Speed test data
# batch_size = 1
# data = [
#     {"messages": [{"role": "user", "content": "请写一遍5000字的综述，介绍开源的搜索引擎项目"}]} for _ in range(batch_size)
# ]

start_time = time.perf_counter()
outputs = model.generate(datas)
total_token = 0
for output in zip(*outputs):
    print([output[0]])
    print(f'output tokens: {output[1]}\n')
    total_token += output[1]
print(f'total tokens: {total_token}, {total_token / (time.perf_counter()-start_time)} token/s')

########################### Speed test result ###########################
# GPU: 4090, gpu_memory_utilization: 0.65
# batch = 1: 91 token/s
# batch = 2: 166 token/s
# batch = 4: 306 token/s
# batch = 8: 491 token/s
# batch = 16: 643 token/s
# batch = 32: 656 token/s