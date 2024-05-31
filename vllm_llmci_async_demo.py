from aiohttp import ClientSession
import time
import asyncio

async def request(url):
    async with ClientSession() as session:
        data = {
            "messages": [
                {"role": "user", "content": "请写一遍5000字的综述，介绍开源的搜索引擎项目"}
            ]
        }
        req_data = {
            "data": data,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "max_new_tokens": 256,
            "stop_token_ids": [151645, 151643],
            "use_beam_search": False,
            "best_of": None,
        }
        async with session.post(url, json=req_data) as response:
            res = await response.json()
            return res

async def main():
    url = 'http://0.0.0.0:9999/worker_generate'
    task_list = []
    for i in range(2):
        task = asyncio.create_task(request(url))
        task_list.append(task)
    done, pending = await asyncio.wait(task_list, timeout=None)
    # 得到执行结果
    for done_task in done:
        print(f"return content: {done_task.result()}\n")

# asyncio.run(main())
loop = asyncio.get_event_loop()
loop.run_until_complete(main())