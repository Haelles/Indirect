import json
import os
import sys
import re
import spacy
import torch
import random
import itertools
import statistics 
import numpy as np
import torch.nn.functional as F
sys.path.append('..') 
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from ConstrainedBeamSearch import *
from tfidf_en import *
from utils import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  


from spacy.tokens import Token  
import inflect  
inflector = inflect.engine()  

def inflect_word(token, pos_tag):  
    if pos_tag == "NN":
        return inflector.singular_noun(token.text) or token.text  
    return token.text  

Token.set_extension("inflect", method=inflect_word) 

p = "" # Please specify the file path.
idf_freq = load_json(f"{p}/datasets/en/idf.json")
median_idf = statistics.median(list(idf_freq.values()))
nlp = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()

commands = []
degrees = []
sensitives = []
with open(f"{p}/datasets/en/commands.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        commands.append(line.strip("\n"))
with open(f"{p}/datasets/en/degrees.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        degrees.append(line.strip("\n"))
with open(f"{p}/datasets/en/sensitives.json", "r", encoding="utf-8") as f:
    sensitives_f = json.load(f)
    for words in sensitives_f.values():
        sensitives += words

def WordExpansion(word):
    doc = nlp(word)
    token = doc[0] 
    singular = token._.inflect('NN') or word
    plural = token._.inflect('NNS') or word 
    present = token._.inflect('VBZ') or word  
    past = token._.inflect('VBD') or word  
    past_participle = token._.inflect('VBN') or word  
    gerund = token._.inflect('VBG') or word
    return list({singular, plural, present, past, past_participle, gerund})

def WordReduction(words):
    reduced_words = [] 
    sorted_words = sorted(list(set(words)), key=len, reverse=False)
    for word in sorted_words:
        if all([u not in word for u in reduced_words]):
            reduced_words.append(word)
    return reduced_words

def generate_direct_or_indirect_requests(num=10, model_names=["Llama", "Mistral"]):
    with open(f"{p}/datasets/en/Dataset.json", encoding="utf-8") as f:
        all_data = json.load(f)
    direct_requests = []
    indirect_requests = []
    for item in all_data:
        item_direct_requests = []
        item_indirect_requests = []
        for _ in range(num):
            indirect_prompt = item["topic"] + item["indirect_template"] if "instruction" not in item.keys() else \
                item["topic"] + item["instruction"] + item["indirect_template"]
            direct_prompt = item["topic"] + item["direct_template"] if "instruction" not in item.keys() else \
                item["topic"] + item["instruction"] + item["direct_template"]
            item_indirect_requests.append(
                {
                    "category" : item["category"],
                    "request" : indirect_prompt,
                    "constraints" : []
                }
            )
            item_direct_requests.append(
                {
                    "category" : item["category"],
                    "request" : direct_prompt,
                    "constraints" : []
                }
            )
        direct_requests.append(item_direct_requests)
        indirect_requests.append(item_indirect_requests)
    for model_name in model_names:
        if not os.path.exists(f"./outputs/en/loop-1/requests/{model_name}"):
            os.makedirs(f"./outputs/en/loop-1/requests/{model_name}")
        with open(f"./outputs/en/loop-1/requests/{model_name}/data_direct.json", "w", encoding="utf-8") as f:
            json.dump(direct_requests, f, ensure_ascii=False, indent=4)
        with open(f"./outputs/en/loop-1/requests/{model_name}/data_indirect.json", "w", encoding="utf-8") as f:
            json.dump(indirect_requests, f, ensure_ascii=False, indent=4)


def load_direct_indirect(loop, model_name):
    all_data = {}
    with open(f"./outputs/en/loop-{loop}/requests/{model_name}/data_direct.json", encoding="utf-8") as f:
        all_data["direct"] = json.load(f)
    with open(f"./outputs/en/loop-{loop}/requests/{model_name}/data_indirect.json", encoding="utf-8") as f:
        all_data["indirect"] = json.load(f)
    return all_data

def load_model(model_name, path="./huggingface_model/"):
    if model_name == "Mistral":
        model_path = os.path.join(path, "Mistral-7B-Instruct-v0.3")
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16).cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        model = model.eval()
    elif model_name == "Llama":
        model_path = os.path.join(path, "Llama-3-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16).cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        model = model.eval()
    return model, tokenizer

def keep_last_sentence(text):
    match = re.search(r'[.!?。！？]', text[::-1])
    if match:
        index = len(text) - match.start() - 1
        return text[:index+1]
    else:
        return text

def generate_answer_baseline(all_data, model_name, model, tokenizer, loop_num):
    response = defaultdict(list)
    for type_, data in all_data.items():
        folder_name = f"./outputs/en/loop-{loop_num}/responses/{model_name}/{type_}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        try:
            with open(os.path.join(folder_name, "baseline.json"), "r", encoding="utf-8") as f:
                response[type_] = json.load(f)
            k = len(response[type_])    
        except:
            k = 0
        for i, items in enumerate(tqdm(data)):
            if i < k :
                continue 
            output_list = []
            for item in items:
                request = item["request"]
                category = item["category"]
                tem = 1.2
                while True:
                    try:
                        if model_name == "Mistral":
                            messages = [
                                {
                                    "role" : "user",
                                    "content" : request,
                                }
                            ]
                            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
                            inputs = {"input_ids":input_ids}
                            num_request_tokens = inputs["input_ids"].shape[1]
                            outputs = model.generate(inputs["input_ids"].cuda(), eos_token_id=tokenizer.eos_token_id, num_beams=10, num_return_sequences=10, do_sample=True, top_p=0.95, temperature=tem, 
                                 max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)[:, num_request_tokens:]
                            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        elif model_name == "Llama":
                            messages = [
                                {
                                    "role" : "user",
                                    "content" : request,
                                }
                            ]
                            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
                            inputs = {"input_ids":input_ids}
                            num_request_tokens = inputs["input_ids"].shape[1]
                            outputs = model.generate(inputs["input_ids"].cuda(), eos_token_id=tokenizer.eos_token_id, num_beams=10, num_return_sequences=10, do_sample=True, top_p=0.95, temperature=tem, 
                                 max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)[:, num_request_tokens:]
                            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        output_list.append(
                            {
                                "category": category,
                                "request": request,
                                "response":keep_last_sentence(random.choice(outputs)),
                            }
                        )
                        break
                    except:
                        tem -= 0.1
                        tem = max(tem, 1.0)
            response[type_].append(output_list)
            folder_name = f"./outputs/en/loop-{loop_num}/responses/{model_name}/{type_}"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            with open(os.path.join(folder_name, "baseline.json"), "w", encoding="utf-8") as f:
                json.dump(response[type_], f, ensure_ascii=False, indent=4)



def generate_answer_BS(all_data, model_name, model, tokenizer, loop_num):
    response = defaultdict(list)
    for type_, data in all_data.items():
        folder_name = f"./outputs/en/loop-{loop_num}/responses/{model_name}/{type_}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        try:
            with open(os.path.join(folder_name, "BS.json"), "r", encoding="utf-8") as f:
                response[type_] = json.load(f)
            k = len(response[type_])
        except:
            k = 0
        for i, items in tqdm(enumerate(data)):
            if i < k :
                continue       
            output_list = []
            for item in items:
                request = item["request"]
                category = item["category"]
                tem = 1.2
                num_beams = 10
                # 也可以设置`stable`，但很慢，但是绝对不会失效
                method = "fast" # 快
                if category == "circumlocution":
                    constraints = [copy.deepcopy(items[0]["constraints"])]
                    constraints[0] += [word.capitalize() for word in constraints[0]]
                else:
                    constraints = [items[0]["constraints"] + commands + degrees + sensitives]
                    constraints[0] += [word.capitalize() for word in constraints[0]]
                if constraints == [[]]:
                    constraints = []
                else:
                    while "" in constraints[0]:
                        constraints[0].remove("")
                while True:
                    try:
                        if model_name == "Mistral":
                            outputs = BeamSearchWarpperMistral(request, model, tokenizer, num_beams=num_beams, num_returns=10, bad_words_list=constraints, do_sample=True, top_p=0.95, temperature=tem, max_length=128, method=method, early_stopping=True, use_cache=True)[0]
                        elif model_name == "Llama":
                            outputs = BeamSearchWarpperLlama(request, model, tokenizer, num_beams=num_beams, num_returns=10, bad_words_list=constraints, do_sample=True, top_p=0.95, temperature=tem, max_length=128, method=method, early_stopping=True, use_cache=True)[0]
                        output_list.append(
                            {
                                "category": category,
                                "request": request,
                                "response":keep_last_sentence(random.choice(outputs)),
                                "constraints" : item["constraints"]
                            }
                        )
                        break
                    except:
                        num_beams += 20
            response[type_].append(output_list)
            folder_name = f"./outputs/en/loop-{loop_num}/responses/{model_name}/{type_}"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            with open(os.path.join(folder_name, "BS.json"), "w", encoding="utf-8") as f:
                json.dump(response[type_], f, ensure_ascii=False, indent=4)

def get_merged_tfidf_result(loop_num=0, merged_models=None, normalization=None, DWM=None):
    assert normalization in [None, "linf", "l1", "l2"]
    assert DWM in [None, "lp", "prob"]  # \(L_p\) denotes performing regularization in \(p\) dimensions, where the value of \(p\) can be 1 or 2.
    directory = os.path.join(f'./outputs/en', f"loop-{loop_num}", "responses")
    models = os.listdir(directory) if merged_models is None else merged_models
    all_direct_items = defaultdict(list)
    all_indirect_items = defaultdict(list)
    # 加载json文件，整合json文件
    for model in models:
        with open(os.path.join(directory, model, "direct", "BS.json"), 'r', encoding="utf-8") as f:
            direct_items = json.load(f)
        for i, items in enumerate(direct_items):
            all_direct_items[i] += items
        with open(os.path.join(directory, model, "indirect", "BS.json"), 'r', encoding="utf-8") as f:
            indirect_items = json.load(f)
        for i, items in enumerate(indirect_items):
            all_indirect_items[i] += items
    # all_..._items : {i : ["request":, "response":]}
    scores = get_merged_tfidf_result_pretrained(all_direct_items, all_indirect_items, normalization, DWM)
    if DWM is None:
        path = f"./outputs/en/loop-{loop_num}/constraints_select/pretrained/rank"
    else:
        path = f"./outputs/en/loop-{loop_num}/constraints_select/pretrained/DWM/{DWM}/{normalization}" if normalization else f"./outputs/en/loop-{loop_num}/constraints_select/pretrained/DWM/{DWM}"
    if not os.path.exists(path):
        os.makedirs(path)
    if merged_models is None:
        with open(os.path.join(path, "merge.json"), "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)
    else:
        with open(os.path.join(path, f"{merged_models[0]}.json"), "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)

def compute_rank_difference(all_direct_tfidf, all_indirect_tfidf):
    computed_rank = []
    for direct_tfidf, indirect_tfidf in zip(all_direct_tfidf, all_indirect_tfidf):
        computed_rank_i = {}
        words = set().union(list(direct_tfidf.keys()), list(indirect_tfidf.keys()))
        for word in words:
            direct_rank = list(direct_tfidf.keys()).index(word)  if word in direct_tfidf.keys() else len(direct_tfidf)
            indirect_rank = list(indirect_tfidf.keys()).index(word) if word in indirect_tfidf.keys() else len(indirect_tfidf) 
            score = direct_rank - indirect_rank
            computed_rank_i[word] = score
        computed_rank_i =  dict(sorted(computed_rank_i.items(), key=lambda x:x[1], reverse=False))
        computed_rank_i_in_indirect = {word:tfidf for word, tfidf in computed_rank_i.items() if word in indirect_tfidf.keys()}
        computed_rank.append((computed_rank_i, computed_rank_i_in_indirect))
    return computed_rank

def DWM_score(tfidf_dict, DWM):
    """
    tfidf : {word:tfidf}
    """
    words = list(tfidf_dict.keys())
    tfidf = np.array(list(tfidf_dict.values()))
    if DWM == "lp":
        tfidf = np.log(tfidf)
    elif DWM == "prob":
        tfidf = np.log(1-np.exp(-tfidf))
    return dict(zip(words, tfidf))

def compute_DWM_difference(all_direct_tfidf, all_indirect_tfidf, DWM, default_value="min"):
    """
    DWM: lp, prob
    """
    computed_DWM = []
    for direct_tfidf, indirect_tfidf in zip(all_direct_tfidf, all_indirect_tfidf):
        direct_DWM = DWM_score(direct_tfidf, DWM)
        indirect_DWM = DWM_score(indirect_tfidf, DWM)
        computed_DWM_i = {}
        if default_value == "min":
            direct_default = min(direct_DWM.values())
            indirect_default = min(indirect_DWM.values())
        elif default_value == "median":
            direct_default = statistics.median(list(direct_DWM.values()))
            indirect_default = statistics.median(list(indirect_DWM.values()))
        words = set().union(list(direct_DWM.keys()), list(indirect_DWM.keys()))
        for word in words:
            direct_score = direct_DWM.get(word, direct_default)
            indirect_score = indirect_DWM.get(word, indirect_default)
            score = direct_score - indirect_score
            computed_DWM_i[word] = score
        computed_DWM_i = dict(sorted(computed_DWM_i.items(), key=lambda x:x[1], reverse=True))
        computed_DWM_i_in_indirect = {word:score for word, score in computed_DWM_i.items() if word in indirect_tfidf.keys()}
        computed_DWM.append((computed_DWM_i, computed_DWM_i_in_indirect))
    return computed_DWM

    
def get_merged_tfidf_result_pretrained(all_direct_items, all_indirect_items, normalization=None, DWM=None):
    all_direct_text = {}
    all_indirect_text = {}
    for i, items in all_direct_items.items():
        all_direct_text[i] = "".join([item["response"] for item in items])
    for i, items in all_indirect_items.items():
        all_indirect_text[i] = "".join([item["response"] for item in items])
    all_direct_tfidf = []
    all_indirect_tfidf = []
    for _, text in all_direct_text.items():
        all_direct_tfidf.append(pretrained_tfidf(text, idf_freq, median_idf, nlp, normalization=normalization, DWM=DWM))
    for _, text in all_indirect_text.items():
        all_indirect_tfidf.append(pretrained_tfidf(text, idf_freq, median_idf, nlp, normalization=normalization, DWM=DWM))
    if DWM is None:
        compute_rank = compute_rank_difference(all_direct_tfidf, all_indirect_tfidf)
        return compute_rank
    else:
        compute_DWM_score = compute_DWM_difference(all_direct_tfidf, all_indirect_tfidf, DWM)
        return compute_DWM_score
    
def logsigmoid(x):
    return -np.log((1+np.exp(-x)))

def add_sentiment_score(file, out_file, max_alpha=1):
    with open(file, "r", encoding="utf-8") as f:
        tfidf_scores = json.load(f)
    new_tfidf_scores = []
    for i, (direct_scores, indirect_scores) in tqdm(enumerate(tfidf_scores)):
        if i < 50 or i > 84:
            alpha = max_alpha
        else:
            alpha = 0
        words = list(direct_scores.keys())
        for word in words:
            score = np.clip(analyzer.polarity_scores(word)["compound"]/2+0.5, 1e-3, 1-1e-3) # P(POS)
            direct_scores[word] = logsigmoid(-direct_scores[word]) + alpha*np.log(score)
            if word in indirect_scores.keys():
                indirect_scores[word] = logsigmoid(-indirect_scores[word]) + alpha*np.log(score)
        direct_scores = dict(sorted(direct_scores.items(), key=lambda x:x[1], reverse=False))
        indirect_scores = dict(sorted(indirect_scores.items(), key=lambda x:x[1], reverse=False))
        new_tfidf_scores.append([direct_scores, indirect_scores])
    if not os.path.exists(out_file):
        os.makedirs(out_file)
    with open(os.path.join(out_file, f"{max_alpha}.json"), "w", encoding="utf-8") as f:
        json.dump(new_tfidf_scores, f, ensure_ascii=False, indent=4)


def add_constraints(loop, words_path, model_name, topK=5):
    with open(words_path, "r", encoding="utf-8") as f:
        words = json.load(f)
    with open(f"./outputs/en/loop-{loop}/requests/{model_name}/data_direct.json", "r", encoding="utf-8") as f:
        data_direct = json.load(f)
    with open(f"./outputs/en/loop-{loop}/requests/{model_name}/data_indirect.json", "r", encoding="utf-8") as f:
        data_indirect = json.load(f)
    new_data_direct = []
    new_data_indirect = []
    for direct_items, indirect_items, item_words in zip(data_direct, data_indirect, words):
        new_item_data_direct = []
        new_item_data_indirect = []
        new_add_words = list(set(list(item_words[0].keys())[:topK]+list(item_words[1].keys())[:topK]))
        new_add_words = WordReduction(list(set(itertools.chain.from_iterable([WordExpansion(word) for word in  new_add_words]))))
        for item in direct_items:
            item["constraints"] += new_add_words
            item["constraints"] = list(set(item["constraints"]))
            if " Note that you are not allowed to use the following words: " not in item["request"]:
                item["request"] += " Note that you are not allowed to use the following words: " + ", ".join(item["constraints"]) + "."
            else:
                item["request"] = item["request"].split(" Note that you are not allowed to use the following words: ")[0]
                item["request"] += " Note that you are not allowed to use the following words: " + ", ".join(item["constraints"]) + "."
            new_item_data_direct.append(item)
        for item in indirect_items:
            item["constraints"] += new_add_words
            item["constraints"] = list(set(item["constraints"]))
            if " Note that you are not allowed to use the following words: " not in item["request"]:
                item["request"] += " Note that you are not allowed to use the following words: " + ", ".join(item["constraints"]) + "."
            else:
                item["request"] = item["request"].split(" Note that you are not allowed to use the following words: ")[0]
                item["request"] += " Note that you are not allowed to use the following words: " + ", ".join(item["constraints"]) + "."
            new_item_data_indirect.append(item)
        new_data_direct.append(new_item_data_direct)
        new_data_indirect.append(new_item_data_indirect)
    path = f"./outputs/en/loop-{loop+1}/requests/{model_name}"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, "data_direct.json"), "w", encoding="utf-8") as f:
       json.dump(new_data_direct, f, indent=4, ensure_ascii=False)
    with open(os.path.join(path, "data_indirect.json"), "w", encoding="utf-8") as f:
       json.dump(new_data_indirect, f, indent=4, ensure_ascii=False)

    