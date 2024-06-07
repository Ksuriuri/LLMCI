from packaging import version
import importlib.metadata
import importlib.util

# Referring to transformers.
def _is_package_available(pkg_name: str, min_version: str):
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
            if version.parse(package_version) >= version.parse(min_version):
                return True
        except importlib.metadata.PackageNotFoundError as ex:
            print(ex)
        return False
    else:
        return False

if _is_package_available("transformers", "4.40.0"):
    from transformers.generation.stopping_criteria import EosTokenCriteria
elif _is_package_available("transformers", "4.38.0"):
    import torch.distributed as dist
else:
    raise RuntimeError(f"Need transformers >= 4.38.0")

from transformers import AutoModelForCausalLM, AutoTokenizer  # , Qwen2ForCausalLM
import torch
from torch import nn
from typing import List, Optional, Union
import time
import warnings

from transformers.generation.logits_process import (
    LogitsProcessorList
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria
)
from transformers.generation.utils import (
    GenerateDecoderOnlyOutput, 
    GenerateEncoderDecoderOutput, 
    GenerateNonBeamOutput
)
from transformers.generation.streamers import BaseStreamer


class TransformersLLMci:
    def __init__(self, tokenizers_path, model_path, max_new_tokens):
        self.device = "cuda"
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizers_path, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
        if _is_package_available("transformers", "4.40.0"):
            self.model._sample = self._sample  # transformers>=4.40
        elif _is_package_available("transformers", "4.38.0"):
            self.model.sample = self.sample

        self.add_stop_char_list = []
        self.fixed_content_list = []
        self.llmci_flag = []

    def generate(self, datas):
        # Empty llmci dict
        self.add_stop_char_list = []
        self.fixed_content_list = []
        self.llmci_flag = []
        outputs_llmci_bos = []  # this list is for tokens that add to the beginning

        texts = []
        for data in datas:
            messages = data["messages"]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)

            # Check llmci input in data
            data['add_stop_char'] = data['add_stop_char'] if 'add_stop_char' in data else []
            data['fixed_content'] = data['fixed_content'] if 'fixed_content' in data else []
            assert isinstance(data['add_stop_char'], list) and isinstance(data['fixed_content'], list), "`add_stop_char` and `fixed_content` must in list type"
            assert len(data['add_stop_char']) == len(data['fixed_content']), "len of `add_stop_char` and `fixed_content` are not equal"

            # In case that add tokens at beggining
            outputs_llmci_bos.append("")
            if data['add_stop_char'] and data['add_stop_char'][0] == '<|llmci_bos|>':
                texts[-1] += data['fixed_content'][0]
                outputs_llmci_bos[-1] += data['fixed_content'][0]
                data['add_stop_char'].pop(0)
                data['fixed_content'].pop(0)

            # Prepare llmci dict
            self.add_stop_char_list.append(data["add_stop_char"])
            # self.fixed_content_list.append(
            #     [torch.tensor(self.tokenizer.encode(str_), dtype=torch.long, device=self.device) 
            #      for str_ in data["fixed_content"]] if data["fixed_content"] else [])
            self.fixed_content = []
            for str_ in data['fixed_content']:
                if str_ == "<|llmci_eos|>" and self.tokenizer.eos_token is not None:
                    str_ = self.tokenizer.eos_token
                self.fixed_content.append(torch.tensor(self.tokenizer.encode(str_), dtype=torch.long, device=self.device))
            self.fixed_content_list.append(self.fixed_content)
            self.llmci_flag.append(False)

        model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=self.max_new_tokens,
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # Calculate tokens
        try:
            output_tokens_num = []
            for g_ids in generated_ids:
                special_tokens_num = 0
                for id_ in g_ids:
                    if id_.item() == self.model.generation_config.pad_token_id:
                        special_tokens_num += 1
                output_tokens_num.append(g_ids.shape[0] - special_tokens_num)
        except Exception as ex:
            print(ex)
            output_tokens_num = [0 for _ in range(len(generated_ids))]

        output_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        output_texts = [output_llmci_bos + output_text for output_llmci_bos, output_text in zip(outputs_llmci_bos, output_texts)]
        return output_texts, output_tokens_num

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.model.generation_config.pad_token_id
        # eos_token_id = eos_token_id if eos_token_id is not None else self.model.generation_config.eos_token_id
        if eos_token_id is not None:
            print(
                "`eos_token_id` is deprecated in this function and will be removed in v4.41, use",
                " `stopping_criteria=StoppingCriteriaList([EosTokenCriteria(eos_token_id=eos_token_id)])` instead.",
                " Otherwise make sure to set `model.generation_config.eos_token_id`"
            )
            stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
        else:
            # TODO remove when the method is totally private
            # need to get `eos_token_id` and add stopping criteria, so that generation does not go forever
            eos_token_id = [
                criteria.eos_token_id.tolist() for criteria in stopping_criteria if hasattr(criteria, "eos_token_id")
            ]
            eos_token_id = eos_token_id[0] if eos_token_id else None
            if eos_token_id is None and self.model.generation_config.eos_token_id is not None:
                eos_token_id = self.model.generation_config.eos_token_id
                stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.model.generation_config.output_scores
        output_logits = output_logits if output_logits is not None else self.model.generation_config.output_logits
        output_attentions = (
            output_attentions if output_attentions is not None else self.model.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.model.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        while self.model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self.model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.model.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.model.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.model.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # llmci replacement
            for idx, fixed_content in enumerate(self.fixed_content_list):
                if self.llmci_flag[idx]:
                    next_tokens[idx] = fixed_content[0][0]
                    self.fixed_content_list[idx][0] = fixed_content[0][1:]
                    if not self.fixed_content_list[idx][0].shape[0]:
                        self.fixed_content_list[idx].pop(0)
                        self.add_stop_char_list[idx].pop(0)
                        self.llmci_flag[idx] = False

            # llmci judge: if encounter stop character
            batch_new_str = self.tokenizer.batch_decode([next_tokens], skip_special_tokens=True)
            for idx, new_str in enumerate(batch_new_str):
                if self.add_stop_char_list[idx] and self.add_stop_char_list[idx][0] in new_str:
                    self.llmci_flag[idx] = True

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self.model._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.model.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.model.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids
        
    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.model.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.model.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.model.generation_config.output_scores
        output_logits = output_logits if output_logits is not None else self.model.generation_config.output_logits
        output_attentions = (
            output_attentions if output_attentions is not None else self.model.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.model.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self.model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.model.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.model.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.model.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # llmci replacement
            for idx, fixed_content in enumerate(self.fixed_content_list):
                if self.llmci_flag[idx]:
                    next_tokens[idx] = fixed_content[0][0]
                    self.fixed_content_list[idx][0] = fixed_content[0][1:]
                    if not self.fixed_content_list[idx][0].shape[0]:
                        self.fixed_content_list[idx].pop(0)
                        self.add_stop_char_list[idx].pop(0)
                        self.llmci_flag[idx] = False

            # llmci judge: if encounter stop character
            batch_new_str = self.tokenizer.batch_decode([next_tokens], skip_special_tokens=True)
            for idx, new_str in enumerate(batch_new_str):
                if self.add_stop_char_list[idx] and self.add_stop_char_list[idx][0] in new_str:
                    self.llmci_flag[idx] = True

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self.model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.model.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids
