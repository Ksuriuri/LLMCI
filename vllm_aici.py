from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from vllm.lora.request import LoRARequest
from vllm.sequence import (Sequence, Logprob,
                           SequenceGroup, SequenceGroupOutput, SequenceOutput,
                           SequenceStatus)

from typing import List, Tuple
import time


class VllmAici:
    def __init__(self,
                 model_path: str,
                 tokenizers_path: str,
                 generation_config,
                 lora_path: str=None,
                 gpu_memory_utilization: float=0.8,
                 max_model_len: int=1024):
        self.lora_path = lora_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizers_path)

        start_time = time.perf_counter()
        self.sampling_params = SamplingParams(**generation_config)
        self.llm = LLM(model=model_path, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len, enable_lora=True if lora_path else False)

        # Replace `_process_sequence_group_outputs` in llm_engine
        self.llm.llm_engine._process_sequence_group_outputs = self._process_sequence_group_outputs

        print('llm load time', time.perf_counter()-start_time)

        self.add_stop_char_dict = {}
        self.fixed_content_dict = {}
        self.aici_flag = {}
    
    def generate(self, datas):
        '''
        example:
            datas = [
                {
                    "messages": [
                        {"role": "system", "content": "system prompt"},
                        {"role": "user", "content": "prompt"}
                    ],
                    "add_stop_char": ['<|aici_bos|>', '\n', '\n', '\n'],
                    "fixed_content": ['1. ', '2. ', '3. ', '<|im_end|>']
                },
                ...
            ]
        '''
        
        # Empty aici dict
        self.add_stop_char_dict = {}
        self.fixed_content_dict = {}
        self.aici_flag = {}

        # Prepare input
        outputs_aici_bos = []  # this list is for tokens that add to the beginning
        texts = []
        seq_id = self.llm.llm_engine.seq_counter.counter  # current seq_id of llm_engine
        for idx, data in enumerate(datas):
            # Extract and compose prompt, note that if role of "system" not in data["messages"],
            # it will add a default system prompt of "You are a helpful assistant."
            text = self.tokenizer.apply_chat_template(
                            data["messages"],
                            tokenize=False,
                            add_generation_prompt=True
            )
            texts.append(text)

            # Since `Sequences` are created in order in vllm, `Sequence.seq_id` increments sequentially. 
            # Therefore, using current seq_id + data index (idx) should correspond to Sequence.seq_id.
            str_idx = str(seq_id+idx)
            # Prepare aici dict
            outputs_aici_bos.append("")
            if 'add_stop_char' in data and 'fixed_content' in data:
                assert isinstance(data['add_stop_char'], list) and isinstance(data['fixed_content'], list), "`add_stop_char` and `fixed_content` must in list type"
                assert len(data['add_stop_char']) == len(data['fixed_content']), "len of `add_stop_char` and `fixed_content` are not equal"
                assert len(data['add_stop_char']) and len(data['fixed_content']), "len of `add_stop_char` and `fixed_content` must > 0"
                if data['add_stop_char'][0] == '<|aici_bos|>':
                    texts[-1] += data['fixed_content'][0]
                    outputs_aici_bos[-1] += data['fixed_content'][0]
                    data['add_stop_char'].pop(0)
                    data['fixed_content'].pop(0)
                self.add_stop_char_dict[str_idx] = data['add_stop_char']
                self.fixed_content_dict[str_idx] = [self.tokenizer.encode(str) for str in data['fixed_content']]
            else:
                self.add_stop_char_dict[str_idx] = []
                self.fixed_content_dict[str_idx] = []
            self.aici_flag[str_idx] = False

        # Inference
        start_time = time.perf_counter()
        print(texts)
        if self.lora_path:
            outputs = self.llm.generate(texts, self.sampling_params, lora_request=LoRARequest("adapter", 1, self.lora_path))
        else:
            outputs = self.llm.generate(texts, self.sampling_params)
        print('batch use time', time.perf_counter()-start_time)
        
        outputs_text = [output_aici_bos + output.outputs[0].text for output, output_aici_bos in zip(outputs, outputs_aici_bos)]
        output_token_nums = [len(output.outputs[0].token_ids) for output in outputs]
        return (outputs_text, output_token_nums,)
    
    def _process_sequence_group_outputs(self, seq_group: SequenceGroup,
                                        outputs: SequenceGroupOutput) -> None:
        # Process prompt logprobs
        prompt_logprobs = outputs.prompt_logprobs
        if prompt_logprobs is not None:
            self.llm.llm_engine.detokenizer.decode_prompt_logprobs_inplace(
                seq_group, prompt_logprobs)
            seq_group.prompt_logprobs = prompt_logprobs

        # Process samples
        samples = outputs.samples
        parent_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        existing_finished_seqs = seq_group.get_finished_seqs()
        parent_child_dict = {
            parent_seq.seq_id: []
            for parent_seq in parent_seqs
        }

        # aici judge: if encounter stop character
        for parent in parent_seqs:
            if len(parent.data.output_token_ids):
                output_str = self.tokenizer.decode(parent.data.output_token_ids[-1], skip_special_tokens=True)
                if len(self.add_stop_char_dict[str(parent.seq_id)]) and self.add_stop_char_dict[str(parent.seq_id)][0] in output_str:
                    self.aici_flag[str(parent.seq_id)] = True

        skip_id = []
        for sample in samples:
            seq_id_ = str(sample.parent_seq_id)
            # Continue if there are multiple child samples.
            if seq_id_ in skip_id:
                continue
            if self.aici_flag[seq_id_]:
                sample.output_token = self.fixed_content_dict[seq_id_][0][0]
                new_logprob = Logprob(logprob=0.0, rank=1)
                sample.logprobs = {sample.output_token: new_logprob}

                self.fixed_content_dict[seq_id_][0] = self.fixed_content_dict[seq_id_][0][1:]
                if len(self.fixed_content_dict[seq_id_][0]) == 0:
                    self.add_stop_char_dict[seq_id_].pop(0)
                    self.fixed_content_dict[seq_id_].pop(0)
                    self.aici_flag[seq_id_] = False
                parent_child_dict[sample.parent_seq_id] = [sample]
                skip_id.append(seq_id_)
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
                self.llm.llm_engine.scheduler.free_seq(parent)
                continue
            # Fork the parent sequence if there are multiple child samples.
            for child_sample in child_samples[:-1]:
                new_child_seq_id = next(self.llm.llm_engine.seq_counter)
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
            self.llm.llm_engine.detokenizer.decode_sequence_inplace(seq, seq_group.sampling_params)
            self.llm.llm_engine._check_stop(seq, seq_group.sampling_params)

        # Non-beam search case
        if not seq_group.sampling_params.use_beam_search:
            # For newly created child sequences, add them to the sequence group
            # and fork them in block manager if they are not finished.
            for seq, parent in child_seqs:
                if seq is not parent:
                    seq_group.add(seq)
                    self.add_stop_char_dict[str(seq.seq_id)] = self.add_stop_char_dict[str(parent.seq_id)]
                    self.fixed_content_dict[str(seq.seq_id)] = self.fixed_content_dict[str(parent.seq_id)]
                    self.aici_flag[str(seq.seq_id)] = self.aici_flag[str(parent.seq_id)]
                    if not seq.is_finished():
                        self.llm.llm_engine.scheduler.fork_seq(parent, seq)

            # Free the finished and selected parent sequences' memory in block
            # manager. Keep them in the sequence group as candidate output.
            # NOTE: we need to fork the new sequences before freeing the
            # old sequences.
            for seq, parent in child_seqs:
                if seq is parent and seq.is_finished():
                    self.llm.llm_engine.scheduler.free_seq(seq)
                    del self.add_stop_char_dict[str(seq.seq_id)]
                    del self.fixed_content_dict[str(seq.seq_id)]
                    del self.aici_flag[str(seq.seq_id)]
            return
        
        # Beam search case
        # Coming soon...


if __name__ == "__main__":
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
