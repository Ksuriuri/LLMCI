"""
此版本对于大量并发更高效
以API的形式调用，使用示例详见vllm_llmci_async_demo.py
"""

# from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from vllm.lora.request import LoRARequest
from vllm.sequence import (Sequence, Logprob,
                           SequenceGroup, SequenceGroupOutput, SequenceOutput,
                           SequenceStatus)
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from typing import List, Tuple
import json
import asyncio
import time
import argparse

app = FastAPI()


class AsyncVllmLLMci:
    def __init__(self,
                 llm_engine: AsyncLLMEngine,
                 tokenizers_path: str,
                 process_num: int=16,
                 lora_path: str=None):
        self.process_num = process_num
        self.lora_path = lora_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizers_path)
        self.llm_engine = llm_engine

        # Replace `_process_sequence_group_outputs` in llm_engine
        self.llm_engine.engine.output_processor._process_sequence_group_outputs = self._process_sequence_group_outputs

        self.add_stop_char_dict = {}
        self.fixed_content_dict = {}
        self.llmci_flag = {}

        self.wait_list = []

    async def generate_stream(self, params):
        data = params.pop("data")
        request_id = params.pop("request_id")
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = params.get("top_k", -1.0)
        presence_penalty = float(params.get("presence_penalty", 0.0))
        frequency_penalty = float(params.get("frequency_penalty", 0.0))
        max_new_tokens = params.get("max_new_tokens", 256)
        stop_token_ids = params.get("stop_token_ids", None) or []
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.append(self.tokenizer.eos_token_id)
        use_beam_search = params.get("use_beam_search", False)
        best_of = params.get("best_of", None)

        request = params.get("request", None)

        # make sampling params in vllm
        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0

        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            use_beam_search=use_beam_search,
            stop_token_ids=stop_token_ids,
            max_tokens=max_new_tokens,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            best_of=best_of,
        )

        # Prepare input
        # Extract and compose prompt, note that if role of "system" not in data["messages"],
        # it will add a default system prompt of "You are a helpful assistant."
        text = self.tokenizer.apply_chat_template(
            data["messages"],
            tokenize=False,
            add_generation_prompt=True
        )

        # Check llmci input in data
        data['add_stop_char'] = data['add_stop_char'] if 'add_stop_char' in data else []
        data['fixed_content'] = data['fixed_content'] if 'fixed_content' in data else []
        assert isinstance(data['add_stop_char'], list) and isinstance(data['fixed_content'], list), "`add_stop_char` and `fixed_content` must in list type"
        assert len(data['add_stop_char']) == len(data['fixed_content']), "len of `add_stop_char` and `fixed_content` are not equal"

        # In case that add tokens at beggining
        outputs_llmci_bos = ""  # this is for tokens that add to the beginning
        if data['add_stop_char'] and data['add_stop_char'][0] == '<|llmci_bos|>':
            text += data['fixed_content'][0]
            outputs_llmci_bos += data['fixed_content'][0]
            data['add_stop_char'].pop(0)
            data['fixed_content'].pop(0)

        # Prepare llmci dict
        self.add_stop_char_dict[request_id] = data['add_stop_char']
        # self.fixed_content_dict[request_id] = [self.tokenizer.encode(str_) for str_ in data['fixed_content']] if data['fixed_content'] else []
        self.fixed_content_dict[request_id] = []
        for str_ in data['fixed_content']:
            if str_ == "<|llmci_eos|>" and self.tokenizer.eos_token is not None:
                str_ = self.tokenizer.eos_token
            self.fixed_content_dict[request_id].append(self.tokenizer.encode(str_))
        self.llmci_flag[request_id] = False

        # Easy Concurrency control
        if hasattr(self.llm_engine, "_request_tracker"):
            if len(self.llm_engine._request_tracker) > self.process_num:
                # 使用列表简单控制并发，使得请求能够尽量保持先到先处理的顺序，极大减少被后来者抢占的概率
                self.wait_list.append(request_id)
                while self.wait_list[0] != request_id or len(self.llm_engine._request_tracker) > self.process_num:
                    await asyncio.sleep(0.05)
                self.wait_list.pop(0)

        # Inference
        if self.lora_path:
            results_generator = self.llm_engine.generate(text, sampling_params, request_id, lora_request=LoRARequest("adapter", 1, self.lora_path))
        else:
            results_generator = self.llm_engine.generate(text, sampling_params, request_id)

        async for request_output in results_generator:
            # prompt = request_output.prompt
            text_outputs = [outputs_llmci_bos + output.text for output in request_output.outputs]
            text_outputs = " ".join(text_outputs)

            aborted = False
            if request and await request.is_disconnected():
                await self.llm_engine.abort(request_id)
                request_output.finished = True
                aborted = True
                for output in request_output.outputs:
                    output.finish_reason = "abort"

            prompt_tokens = len(request_output.prompt_token_ids)
            completion_tokens = sum(
                len(output.token_ids) for output in request_output.outputs
            )
            ret = {
                "output_text": text_outputs,
                "error_code": 0,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "cumulative_logprob": [
                    output.cumulative_logprob for output in request_output.outputs
                ],
                "finish_reason": request_output.outputs[0].finish_reason
                if len(request_output.outputs) == 1
                else [output.finish_reason for output in request_output.outputs],
            }
            # Emit twice here to ensure a 'finish_reason' with empty content in the OpenAI API response.
            # This aligns with the behavior of model_worker.
            if request_output.finished:
                yield (json.dumps({**ret, **{"finish_reason": None}}) + "\0").encode()
            yield (json.dumps(ret) + "\0").encode()

            if aborted:
                break

    async def generate(self, params):
        async for x in self.generate_stream(params):
            pass
        return json.loads(x[:-1].decode())


    def _process_sequence_group_outputs(self, seq_group: SequenceGroup,
                                        outputs: SequenceGroupOutput) -> None:
        # Process prompt logprobs
        prompt_logprobs = outputs.prompt_logprobs
        if prompt_logprobs is not None and \
            seq_group.sampling_params.detokenize and self.llm_engine.engine.output_processor.detokenizer:
            self.llm_engine.engine.output_processor.detokenizer.decode_prompt_logprobs_inplace(
                seq_group, prompt_logprobs)
            seq_group.prompt_logprobs = prompt_logprobs

        # Process samples
        samples = outputs.samples
        parent_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)

        if len(self.add_stop_char_dict[seq_group.request_id]) > 0:
            samples = samples[:1]
            parent_seqs = parent_seqs[:1]

        existing_finished_seqs = seq_group.get_finished_seqs()
        parent_child_dict = {
            parent_seq.seq_id: []
            for parent_seq in parent_seqs
        }

        # llmci judge: if encounter stop character
        for parent in parent_seqs:
            if len(parent.data.output_token_ids):
                output_str = self.tokenizer.decode([parent.data.output_token_ids[-1]], skip_special_tokens=True)
                if len(self.add_stop_char_dict[seq_group.request_id]) and self.add_stop_char_dict[seq_group.request_id][0] in output_str:
                    self.llmci_flag[seq_group.request_id] = True

        for sample in samples:
            if self.llmci_flag[seq_group.request_id]:
                sample.output_token = self.fixed_content_dict[seq_group.request_id][0][0]
                new_logprob = Logprob(logprob=0.0, rank=1)
                sample.logprobs = {sample.output_token: new_logprob}

                self.fixed_content_dict[seq_group.request_id][0] = self.fixed_content_dict[seq_group.request_id][0][1:]
                if len(self.fixed_content_dict[seq_group.request_id][0]) == 0:
                    self.add_stop_char_dict[seq_group.request_id].pop(0)
                    self.fixed_content_dict[seq_group.request_id].pop(0)
                    self.llmci_flag[seq_group.request_id] = False
                parent_child_dict[sample.parent_seq_id] = [sample]
            else:
                parent_child_dict[sample.parent_seq_id].append(sample)
        # List of (child, parent)
        child_seqs: List[Tuple[Sequence, Sequence]] = []

        # Process the child samples for each parent sequence
        for parent in parent_seqs:
            child_samples: List[SequenceOutput] = parent_child_dict[parent.seq_id]
            if len(child_samples) == 0:
                # This parent sequence has no children samples. Remove
                # the parent sequence from the sequence group since it will
                # not be used in the future iterations.
                parent.status = SequenceStatus.FINISHED_ABORTED
                seq_group.remove(parent.seq_id)
                self.llm_engine.engine.output_processor.scheduler.free_seq(parent)
                continue
            # Fork the parent sequence if there are multiple child samples.
            for child_sample in child_samples[:-1]:
                new_child_seq_id: int = next(self.llm_engine.engine.output_processor.seq_counter)
                child = parent.fork(new_child_seq_id)
                child.append_token_id(child_sample.output_token, child_sample.logprobs)
                child_seqs.append((child, parent))
            # Continue the parent sequence for the last child sample.
            # We reuse the parent sequence here to reduce redundant memory
            # copies, especially when using non-beam search sampling methods.
            last_child_sample = child_samples[-1]
            parent.append_token_id(last_child_sample.output_token, last_child_sample.logprobs)
            child_seqs.append((parent, parent))

        for seq, _ in child_seqs:
            if seq_group.sampling_params.detokenize and self.llm_engine.engine.output_processor.detokenizer:
                new_char_count = self.llm_engine.engine.output_processor.detokenizer.decode_sequence_inplace(
                    seq, seq_group.sampling_params)
            else:
                new_char_count = 0
            self.llm_engine.engine.output_processor.stop_checker.maybe_stop_sequence(seq, new_char_count,
                                                                                  seq_group.sampling_params)

        # Non-beam search case
        if not seq_group.sampling_params.use_beam_search:
            # For newly created child sequences, add them to the sequence group
            # and fork them in block manager if they are not finished.
            for seq, parent in child_seqs:
                if seq is not parent:
                    seq_group.add(seq)
                    if not seq.is_finished():
                        self.llm_engine.engine.output_processor.scheduler.fork_seq(parent, seq)

            # Free the finished and selected parent sequences' memory in block
            # manager. Keep them in the sequence group as candidate output.
            # NOTE: we need to fork the new sequences before freeing the
            # old sequences.
            for seq, parent in child_seqs:
                if seq is parent and seq.is_finished():
                    self.llm_engine.engine.output_processor.scheduler.free_seq(seq)
                    del self.add_stop_char_dict[seq_group.request_id]
                    del self.fixed_content_dict[seq_group.request_id]
                    del self.llmci_flag[seq_group.request_id]
            return
        
        # Beam search case
        # Coming soon...


def create_background_tasks(request_id):
    async def abort_request() -> None:
        await engine.abort(request_id)

    background_tasks = BackgroundTasks()
    background_tasks.add_task(abort_request)
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    request_id = random_uuid()
    params["request_id"] = request_id
    params["request"] = request
    generator = worker.generate_stream(params)
    background_tasks = create_background_tasks(request_id)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    request_id = random_uuid()
    params["request_id"] = request_id
    params["request"] = request
    output = await worker.generate(params)
    await engine.abort(request_id)
    return JSONResponse(output)


# @app.post("/test")
# async def api_generate(request: Request):
#     params = await request.json()
#     request_id = random_uuid()
#     params["request_id"] = request_id
#     return JSONResponse(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--model-path", type=str, default="/home/oneway/ssd2t/model/qwen/Qwen1___5-14B-Chat-GPTQ-Int4")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--trust_remote_code",
        action="store_false",
        default=True,
        help="Trust remote code (e.g., from HuggingFace) when"
        "downloading the model and tokenizer.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.86,
        help="The ratio (between 0 and 1) of GPU memory to"
        "reserve for the model weights, activations, and KV cache. Higher"
        "values will increase the KV cache size and thus improve the model's"
        "throughput. However, if the value is too high, it may cause out-of-"
        "memory (OOM) errors.",
    )
    parser.add_argument('--max-total-len', type=int, default=1024)
    parser.add_argument("--log-requests", action="store_true")
    parser.add_argument(
        '--process-num', 
        type=int, 
        default=16,
        help="GPU同时处理的请求数的最大值，4090GPU设置为16能达到较高吞吐率。"
        "此值设置过大会导致频繁调度降低吞吐率，设置过小会导致显存利用率不足，"
        "亦会降低吞吐率。"
    )

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    if args.model_path:
        args.model = args.model_path
    if args.num_gpus > 1:
        args.tensor_parallel_size = args.num_gpus
    if args.max_total_len:
        args.max_model_len = args.max_total_len
    args.disable_log_requests = not args.log_requests

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    worker = AsyncVllmLLMci(
        engine,
        args.model_path,
        args.process_num,
    )
    uvicorn.run(app, host=args.host, port=args.port)  # , log_level="info"


