import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from transformers.generation.beam_search import BeamSearchScorer
from transformers import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, RepetitionPenaltyLogitsProcessor


global model_name 

def torch_int_div(tensor1, tensor2):
    return torch.div(tensor1, tensor2, rounding_mode="floor")

def end_equals_start(str1, str2):
    for i in range(len(str1)):
        substr = str1[i:]
        if str2.startswith(substr):
            return substr
    return ""

def preprocess(inputs, num_beams):
    batch_size, cur_len = inputs["input_ids"].shape
    for tensor_name, tensor in inputs.items():
        tensor = tensor.unsqueeze(1).expand(batch_size, num_beams, cur_len)
        tensor = tensor.reshape(batch_size * num_beams, cur_len)
        inputs[tensor_name] = tensor
    return inputs

def constraint_process(logprob, buffer):
    for beam_id, beam_buffer in buffer.items():
        for word, word_buffer in beam_buffer.items():
            written = word_buffer["written"]
            vocab = word_buffer["vocab"]
            bad_vocab = dict(filter(lambda x: word in written+x[0], vocab.items()))
            logprob[beam_id, torch.tensor(list(bad_vocab.values()), dtype=torch.int64)] = -torch.inf
    return logprob

def scores_process(input_ids, scores, top_k=0, top_p=1.0, temperature=1.0, min_tokens_to_keep=1):
    if top_k > 0:
        top_k_process = TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=min_tokens_to_keep)
        scores = top_k_process(input_ids, scores)
    if top_p < 1.0:
        top_p_process = TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=min_tokens_to_keep)
        scores = top_p_process(input_ids, scores)
    if temperature != 1.0:
        temperature_process = TemperatureLogitsWarper(temperature=temperature)
        scores = temperature_process(input_ids, scores)
    return scores


def update_buffer(next_beam_tokens, next_beam_indices, buffer, tokenizer):
    # next_beam_tokens == 0是这个分支结束的标志
    new_buffer = copy.deepcopy(buffer)
    new_tokens = tokenizer.batch_decode(next_beam_tokens.unsqueeze(-1))
    for beam_id, beam_buffer in buffer.items():
        if next_beam_tokens[beam_id] == 0:
            continue
        for word, _ in beam_buffer.items():
            written = buffer[next_beam_indices[beam_id].item()][word]["written"]
            new_token = new_tokens[beam_id]
            new_written = end_equals_start(written+new_token, word)
            new_buffer[beam_id][word]["written"] = new_written
    return new_buffer


def ban_beams(generated_tokens, beam_scores, bad_words_list, num_beams, next_beam_tokens):
    for beam_id in range(len(beam_scores)):
        if next_beam_tokens[beam_id] == 0:
            continue
        batch_idx = beam_id // num_beams
        for bad_word in bad_words_list[batch_idx]:
            if bad_word in generated_tokens[beam_id]:
                beam_scores[beam_id] = -torch.inf
                break
    return beam_scores

def greedy_search_new_tokens(scores, batch_size, num_beams, vocab_size):
    scores = scores.reshape(batch_size, num_beams * vocab_size)
    new_scores, new_tokens = torch.topk(scores, 2*num_beams, dim=1, largest=True, sorted=True)
    return new_scores, new_tokens

def sample_search_new_tokens(
        input_ids, 
        scores, 
        batch_size, 
        num_beams, 
        vocab_size, 
        top_k=0, 
        top_p=1.0, 
        temperature=1.0):
    scores_processed = scores_process(input_ids, scores, top_k, top_p, temperature, 2*num_beams)
    scores_processed = scores_processed.reshape(batch_size, num_beams*vocab_size)
    probs = F.softmax(scores_processed, dim=-1)
    new_tokens = torch.multinomial(probs, num_samples=2*num_beams)
    new_scores = torch.gather(scores_processed, -1, new_tokens)
    new_scores, new_scores_indices = torch.sort(new_scores, descending=True, dim=1)
    new_tokens = torch.gather(new_tokens, -1, new_scores_indices)
    return new_scores, new_tokens

def past_key_values_beam_process(past_key_values, next_beam_indices):
    global model_name
    new_past_key_values = []
    for layer in past_key_values:
        new_past_key_values_layer = []
        for data in layer:
            if model_name in ["Qwen", "Deepseek", "Yi", "BaiChuan", "Alpaca", "Llama", "Mistral"]:
                new_past_key_values_data =  data[next_beam_indices]
            elif model_name in ["ChatGLM"]:
                new_past_key_values_data =  data[:, next_beam_indices, :, :]
            new_past_key_values_layer.append(new_past_key_values_data)
        new_past_key_values_layer = tuple(new_past_key_values_layer)
        new_past_key_values.append(new_past_key_values_layer)
    del past_key_values
    return tuple(new_past_key_values)

def beam_search(inputs, model, tokenizer, beam_scorer, max_length, bad_words_list=[], num_returns=None,
                do_sample=False, top_k=0, top_p=1.0, temperature=1.0, repetition_penalty=1.0, 
                method="stable", use_cache=False):
    input_ids = inputs["input_ids"].cuda()
    global model_name
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else 1
    if model_name in ["Qwen", "Deepseek", "Yi", "BaiChuan", "Alpaca", "Mistral", "Llama"]:
        eos_token_id = [tokenizer.eos_token_id]
    elif model_name in ["ChatGLM"]:
        eos_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"),
                    tokenizer.get_command("<|observation|>")]
    
    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams
    batch_beam_size, cur_len = input_ids.shape
    if num_returns is None:
        num_returns = num_beams
    
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))
    
    if len(bad_words_list) > 0:
        if method == "stable":
            vocab = tokenizer.get_vocab()
            all_bad_words_vocabs = []
            for batch in range(batch_size):
                batch_bad_words_vocabs = {}
                for word in bad_words_list[batch]:
                    v = dict(filter(lambda x : word[-1] in x[0], vocab.items()))
                    batch_bad_words_vocabs[word] = v
                all_bad_words_vocabs.append(batch_bad_words_vocabs)
            buffer = {
                beam_id:{
                    word:{
                        "written" : "",
                        "vocab":all_bad_words_vocabs[beam_id//num_beams][word]
                    } for word in bad_words_list[beam_id//num_beams]
                } for beam_id in range(batch_beam_size)
            }

    
    if use_cache:
        past_key_values = None
        next_beam_tokens = input_ids

    if repetition_penalty != 1.0:
        RPLP = RepetitionPenaltyLogitsProcessor(repetition_penalty)

    max_length += cur_len
    start_len = cur_len
    iters = 0

    while cur_len < max_length:
        with torch.no_grad():
            
            if use_cache:
                outputs, past_key_values = model(next_beam_tokens, past_key_values=past_key_values, use_cache=True).to_tuple()
                next_token_logits = outputs[:, -1, :]
            else:
                outputs = model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]

            next_token_scores = nn.functional.log_softmax(next_token_logits, dim=-1)
        
            if method == "stable" and len(bad_words_list) > 0: 
                next_token_scores = constraint_process(next_token_scores, buffer)

            if repetition_penalty != 1.0:
                next_token_scores = RPLP(input_ids, next_token_scores)
       
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            vocab_size = next_token_scores.shape[-1]
        
            if do_sample:
                next_token_scores, next_tokens = \
                    sample_search_new_tokens(input_ids, next_token_scores, batch_size, num_beams, vocab_size, top_k, top_p, temperature)
            else:
                next_token_scores, next_tokens = \
                    greedy_search_new_tokens(next_token_scores, batch_size, num_beams, vocab_size) 
            
            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size 

            beam_outputs = beam_scorer.process(input_ids, next_token_scores, next_tokens, next_indices,
                                           pad_token_id=pad_token_id,eos_token_id=eos_token_id)
        
            beam_scores = beam_outputs["next_beam_scores"]
            next_beam_tokens = beam_outputs["next_beam_tokens"]
            next_beam_indices = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[next_beam_indices, :], next_beam_tokens.unsqueeze(-1)], dim=-1)
            generated_tokens = tokenizer.batch_decode(input_ids[:, start_len:], skip_special_tokens=True)


            if len(bad_words_list) > 0:
                if method == "stable":
                    buffer = update_buffer(next_beam_tokens, next_beam_indices, buffer, tokenizer)
                elif method == "fast":
                    beam_scores = ban_beams(generated_tokens, beam_scores, bad_words_list, num_beams, next_beam_tokens)
                    
            next_beam_tokens = next_beam_tokens.unsqueeze(-1)
            if use_cache:
                past_key_values = past_key_values_beam_process(past_key_values, next_beam_indices)    
            
            iters += 1
            cur_len += 1
            if beam_scorer.is_done:
                break
         
    sequence_outputs = beam_scorer.finalize(input_ids, beam_scores, next_tokens, next_indices,
                                            pad_token_id=pad_token_id,eos_token_id=eos_token_id,max_length=max_length)
    return sequence_outputs

def BeamSearchGeneration(inputs, model, tokenizer, max_length, num_beams, num_returns=1, bad_words_list=[], do_sample=False, 
                         top_k=0, top_p=1.0, temperature=1.0, length_penalty=0.0, repetition_penalty=1.0, early_stopping=False, method="stable", use_cache=False):
    
    assert num_returns <= num_beams, "请确保num_returns不超过num_beams！"
    assert method in ["stable", "fast"], "method只能选'stable'或'fast'!"
    inputs = copy.deepcopy(inputs)
    batch_size, cur_len = inputs["input_ids"].shape
    inputs = preprocess(inputs, num_beams)
    beam_scorer = BeamSearchScorer(batch_size, num_beams, device=inputs["input_ids"].device, num_beam_hyps_to_keep=num_returns,
                                  length_penalty=length_penalty, do_early_stopping=early_stopping)
    sequence_outputs = beam_search(inputs, model, tokenizer, beam_scorer, max_length, bad_words_list, 
                                   num_returns, do_sample,top_k, top_p, temperature, repetition_penalty, 
                                   method, use_cache)
    outputs = sequence_outputs["sequences"]
    return outputs

def BeamSearchWarpperChatGLM(texts, model, tokenizer, max_length, num_beams, num_returns=1, bad_words_list=[], do_sample=False, 
                         top_k=0, top_p=1.0, temperature=1.0, length_penalty=1.0, repetition_penalty=1.0, 
                         early_stopping=False, method="stable", use_cache=False):
    global model_name 
    model_name = "ChatGLM"
    if not isinstance(texts, list):
        texts = [texts]
    outputs = []
    bad_words_list += [[] for _ in range(len(texts)-len(bad_words_list))]
    for text, bad_words in zip(texts, bad_words_list):
        bad_words = [bad_words] if bad_words != [] else []
        inputs = tokenizer.build_chat_input(text, history=[], role="user")
        output = BeamSearchGeneration(inputs, model, tokenizer, num_beams=num_beams, num_returns=num_returns, 
                                      bad_words_list=bad_words,  do_sample=do_sample, top_p=top_p, temperature=temperature, 
                                      max_length=max_length, method=method, early_stopping=early_stopping, repetition_penalty=repetition_penalty,
                                      top_k=top_k, length_penalty=length_penalty, use_cache=use_cache)
        output = tokenizer.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        outputs.append(output)
    return outputs

def BeamSearchWarpperDeepseek(texts, model, tokenizer, max_length, num_beams, num_returns=1, bad_words_list=[], do_sample=False, 
                         top_k=0, top_p=1.0, temperature=1.0, length_penalty=1.0, repetition_penalty=1.0, 
                         early_stopping=False, method="stable", use_cache=False):
    global model_name 
    model_name = "Deepseek"
    if not isinstance(texts, list):
        texts = [texts]
    outputs = []
    bad_words_list += [[] for _ in range(len(texts)-len(bad_words_list))]
    for text, bad_words in zip(texts, bad_words_list):
        bad_words = [bad_words] if bad_words != [] else []
        messages = [
            {
                "role" : "user",
                "content" : text,
            }
        ]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = {"input_ids":input_ids}
        output = BeamSearchGeneration(inputs, model, tokenizer, num_beams=num_beams, num_returns=num_returns, 
                                      bad_words_list=bad_words,  do_sample=do_sample, top_p=top_p, temperature=temperature, 
                                      max_length=max_length, method=method, early_stopping=early_stopping, repetition_penalty=repetition_penalty,
                                      top_k=top_k, length_penalty=length_penalty, use_cache=use_cache)
        output = tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)
        outputs.append(output)
    return outputs

def BeamSearchWarpperQwen(texts, model, tokenizer, max_length, num_beams, num_returns=1, bad_words_list=[], do_sample=False, 
                         top_k=0, top_p=1.0, temperature=1.0, length_penalty=1.0, repetition_penalty=1.0, early_stopping=False, method="stable",
                         use_cache=False):
    global model_name 
    model_name = "Qwen"
    if not isinstance(texts, list):
        texts = [texts]
    outputs = []
    bad_words_list += [[] for _ in range(len(texts)-len(bad_words_list))]
    for text, bad_words in zip(texts, bad_words_list):
        bad_words = [bad_words] if bad_words != [] else []
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful assistant."
            },
            {
                "role": "user", 
                "content": text
            }
        ]
        messages = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs = tokenizer([messages], return_tensors="pt")
        output = BeamSearchGeneration(inputs, model, tokenizer, num_beams=num_beams, num_returns=num_returns, 
                                      bad_words_list=bad_words,  do_sample=do_sample, top_p=top_p, temperature=temperature, 
                                      max_length=max_length, method=method, early_stopping=early_stopping, repetition_penalty=repetition_penalty,
                                      top_k=top_k, length_penalty=length_penalty, use_cache=use_cache)
        output = tokenizer.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        outputs.append(output)
    return outputs

def BeamSearchWarpperYi(texts, model, tokenizer, max_length, num_beams, num_returns=1, bad_words_list=[], do_sample=False, 
                         top_k=0, top_p=1.0, temperature=1.0, length_penalty=1.0, early_stopping=False, repetition_penalty=1.0,
                         method="stable", use_cache=False):
    global model_name 
    model_name = "Yi"
    if not isinstance(texts, list):
        texts = [texts]
    outputs = []
    bad_words_list += [[] for _ in range(len(texts)-len(bad_words_list))]
    for text, bad_words in zip(texts, bad_words_list):
        bad_words = [bad_words] if bad_words != [] else []
        messages = [
            {
                "role": "user", 
                "content": text
            }
        ]
        input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        inputs = {"input_ids":input_ids}
        output = BeamSearchGeneration(inputs, model, tokenizer, num_beams=num_beams, num_returns=num_returns, 
                                      bad_words_list=bad_words,  do_sample=do_sample, top_p=top_p, temperature=temperature, 
                                      max_length=max_length, method=method, early_stopping=early_stopping, 
                                      top_k=top_k, length_penalty=length_penalty, use_cache=use_cache, repetition_penalty=repetition_penalty)
        output = tokenizer.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append(output)
    return outputs

def BeamSearchWarpperBaiChuan(texts, model, tokenizer, max_length, num_beams, num_returns=1, bad_words_list=[], do_sample=False, 
                         top_k=0, top_p=1.0, temperature=1.0, length_penalty=1.0, early_stopping=False, repetition_penalty=1.0,
                         method="stable", use_cache=False):
    global model_name 
    model_name = "BaiChuan"
    if not isinstance(texts, list):
        texts = [texts]
    outputs = []
    bad_words_list += [[] for _ in range(len(texts)-len(bad_words_list))]
    for text, bad_words in zip(texts, bad_words_list):
        bad_words = [bad_words] if bad_words != [] else []
        messages = [
            {
                "role": "user", 
                "content": text
            }
        ]
        input_tokens = [model.generation_config.user_token_id] \
                    + tokenizer.encode(messages[0]["content"]) \
                    + [model.generation_config.assistant_token_id]
        input_ids = torch.LongTensor([input_tokens])
        inputs = {"input_ids":input_ids}
        output = BeamSearchGeneration(inputs, model, tokenizer, num_beams=num_beams, num_returns=num_returns, 
                                      bad_words_list=bad_words,  do_sample=do_sample, top_p=top_p, temperature=temperature, 
                                      max_length=max_length, method=method, early_stopping=early_stopping, repetition_penalty=repetition_penalty,
                                      top_k=top_k, length_penalty=length_penalty, use_cache=use_cache)
        output = tokenizer.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append(output)
    return outputs

def BeamSearchWarpperAlpaca(texts, model, tokenizer, max_length, num_beams, num_returns=1, bad_words_list=[], do_sample=False, top_k=0, top_p=1.0, temperature=1.0, length_penalty=1.0, early_stopping=False, repetition_penalty=1.0,
method="stable", use_cache=False):
    global model_name 
    model_name = "Alpaca"
    if not isinstance(texts, list):
        texts = [texts]
    outputs = []
    bad_words_list += [[] for _ in range(len(texts)-len(bad_words_list))]
    for text, bad_words in zip(texts, bad_words_list):
        bad_words = [bad_words] if bad_words != [] else []
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful and honest assistant."
            },
            {
                "role": "user", 
                "content": text,
            },
        ]
        input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        inputs = {"input_ids":input_ids}
        output = BeamSearchGeneration(inputs, model, tokenizer, num_beams=num_beams, num_returns=num_returns, 
                                      bad_words_list=bad_words,  do_sample=do_sample, top_p=top_p, temperature=temperature, 
                                      max_length=max_length, method=method, early_stopping=early_stopping, 
                                      top_k=top_k, length_penalty=length_penalty, use_cache=use_cache, repetition_penalty=repetition_penalty)
        output = tokenizer.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append(output)
    return outputs

def BeamSearchWarpperMistral(texts, model, tokenizer, max_length, num_beams, num_returns=1, bad_words_list=[], do_sample=False, top_k=0, top_p=1.0, temperature=1.0, length_penalty=1.0, early_stopping=False, repetition_penalty=1.0,
method="stable", use_cache=False):
    global model_name 
    model_name = "Mistral"
    if not isinstance(texts, list):
        texts = [texts]
    outputs = []
    bad_words_list += [[] for _ in range(len(texts)-len(bad_words_list))]
    for text, bad_words in zip(texts, bad_words_list):
        bad_words = [bad_words] if bad_words != [] else []
        messages = [
            {
                "role": "user", 
                "content": text
            },
        ]
        input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        inputs = {"input_ids":input_ids}
        output = BeamSearchGeneration(inputs, model, tokenizer, num_beams=num_beams, num_returns=num_returns, 
                                      bad_words_list=bad_words,  do_sample=do_sample, top_p=top_p, temperature=temperature, 
                                      max_length=max_length, method=method, early_stopping=early_stopping, 
                                      top_k=top_k, length_penalty=length_penalty, use_cache=use_cache, repetition_penalty=repetition_penalty)
        output = tokenizer.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append(output)
    return outputs

def BeamSearchWarpperLlama(texts, model, tokenizer, max_length, num_beams, num_returns=1, bad_words_list=[], do_sample=False, top_k=0, top_p=1.0, temperature=1.0, length_penalty=1.0, early_stopping=False, repetition_penalty=1.0,
method="stable", use_cache=False):
    global model_name 
    model_name = "Llama"
    if not isinstance(texts, list):
        texts = [texts]
    outputs = []
    bad_words_list += [[] for _ in range(len(texts)-len(bad_words_list))]
    for text, bad_words in zip(texts, bad_words_list):
        bad_words = [bad_words] if bad_words != [] else []
        messages = [
            {
                "role": "user", 
                "content": text
            },
        ]
        input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        inputs = {"input_ids":input_ids}
        output = BeamSearchGeneration(inputs, model, tokenizer, num_beams=num_beams, num_returns=num_returns, 
                                      bad_words_list=bad_words,  do_sample=do_sample, top_p=top_p, temperature=temperature, 
                                      max_length=max_length, method=method, early_stopping=early_stopping, 
                                      top_k=top_k, length_penalty=length_penalty, use_cache=use_cache, repetition_penalty=repetition_penalty)
        output = tokenizer.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append(output)
    return outputs