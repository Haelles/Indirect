import json
import os
import sys
import re
import torch
import random
import statistics 
import torch.nn.functional as F
sys.path.append('..') 
from tqdm import tqdm
from utils import *
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from ConstrainedBeamSearch import *
from tfidf_cn import *
from jieba.analyse.tfidf import IDFLoader

IDF = IDFLoader(f"./datasets/cn/idf-jieba.txt")
idf_freq, median_idf = IDF.get_idf()
min_idf = min(list(idf_freq.values()))

# derive using HanLP sentiment analysis
with open(f"./datasets/cn/words_sentiment_scores.json", "r", encoding="utf-8") as f:
    words_sentiment_scores = json.load(f)

others = []
commands = []
degrees = []
sensitives = []
with open(f"./datasets/cn/others.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        others.append(line.strip("\n"))
with open(f"./datasets/cn/commands.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        commands.append(line.strip("\n"))
with open(f"./datasets/cn/degrees.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        degrees.append(line.strip("\n"))
with open(f"./datasets/cn/sensitives.json", "r", encoding="utf-8") as f:
    sensitives_f = json.load(f)
    for words in sensitives_f.values():
        sensitives += words


def generate_direct_or_indirect_requests(num=10, model_names=["Yi", "Qwen", "Deepseek", "BaiChuan", "Alpaca"]):
    with open(f"./datasets/cn/Dataset.json", encoding="utf-8") as f:
        Dataset = json.load(f)
    with open(f"./datasets/cn/templates.json", "r", encoding="utf-8") as f:
        templates = json.load(f)
    direct_requests = []
    indirect_requests = []
    for item in Dataset:
        item_direct_requests = []
        item_indirect_requests = []
        if item["category"] == "迂回描述":
            indirect_template = templates["indirect"]["迂回描述"]
        elif item["category"] == "创意写作":
            indirect_template = templates["indirect"]["创意写作"]
        else:
            indirect_template = templates["indirect"]["其他"]
        direct_template = templates["direct"]
        for _ in range(num):
            indirect_prompt = random.choice(item["indirect_template"])
            direct_prompt = random.choice(item["direct_template"])
            indirect_request = indirect_template.replace("<问题>", indirect_prompt)
            direct_request = direct_template.replace("<问题>", direct_prompt)
            item_indirect_requests.append(
                {
                    "category": item["category"],
                    "request" : indirect_request,
                    "constraints" : []
                }
            )
            item_direct_requests.append(
                {
                    "category": item["category"],
                    "request" : direct_request,
                    "constraints" : []
                }
            )
        direct_requests.append(item_direct_requests)
        indirect_requests.append(item_indirect_requests)
    for model_name in model_names:
        if not os.path.exists(f"./outputs/cn/loop-1/requests/{model_name}"):
            os.makedirs(f"./outputs/cn/loop-1/requests/{model_name}")
        with open(f"./outputs/cn/loop-1/requests/{model_name}/data_direct.json", "w", encoding="utf-8") as f:
            json.dump(direct_requests, f, ensure_ascii=False, indent=4)
        with open(f"./outputs/cn/loop-1/requests/{model_name}/data_indirect.json", "w", encoding="utf-8") as f:
            json.dump(indirect_requests, f, ensure_ascii=False, indent=4)


def load_direct_indirect(loop, model_name):
    all_data = {}
    with open(f"./outputs/cn/loop-{loop}/requests/{model_name}/data_direct.json", encoding="utf-8") as f:
        all_data["direct"] = json.load(f)
    with open(f"./outputs/cn/loop-{loop}/requests/{model_name}/data_indirect.json", encoding="utf-8") as f:
        all_data["indirect"] = json.load(f)
    return all_data

def load_model(model_name, path="./huggingface_model/"):
    if model_name == "Qwen":
        model_path = os.path.join(path, "Qwen1.5-7B-Chat")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,  max_split_size_mb=4)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16,
                                             pad_token_id=tokenizer.pad_token_id,device_map="auto").cuda()
        model = model.eval()
    elif model_name == "Deepseek":
        model_path = os.path.join(path, "deepseek-llm-7b-chat")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,  max_split_size_mb=4)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16, 
                                                    device_map="auto").cuda()
        model = model.eval()
        model.generation_config = GenerationConfig.from_pretrained(model_path)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    elif model_name == "Yi":
        model_path = os.path.join(path, "Yi-6B-Chat")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,  max_split_size_mb=4)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16,
                                             pad_token_id=tokenizer.pad_token_id,device_map="auto").cuda()
        model = model.eval()
    elif model_name == "BaiChuan":
        model_path = os.path.join(path, "baichuan2-7B-Chat")
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16).cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        model = model.eval()
    elif model_name == "Alpaca":
        model_path = os.path.join(path, "chinese-alpaca-2-7b")
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16).cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        model = model.eval()
    elif model_name == "Mistral":
        model_path = os.path.join(path, "Mistral-7B-Instruct-v0.3")
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
        for items in tqdm(data):
            output_list = []
            for request in [item["request"] for item in items]:
                tem = 1.2
                while True:
                    try:
                        if model_name == "Deepseek":
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
                                 max_new_tokens=128)[:, num_request_tokens:]
                            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        elif model_name == "Qwen":
                            messages = [
                            {
                                "role": "system", 
                                "content": "You are a helpful assistant."
                            },
                            {
                                "role": "user", 
                                "content": request
                            }
                            ]
                            messages = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                            inputs = tokenizer([messages], return_tensors="pt")
                            num_request_tokens = inputs["input_ids"].shape[1] 
                            outputs = model.generate(inputs["input_ids"].cuda(), eos_token_id=tokenizer.eos_token_id, num_beams=10, num_return_sequences=10, do_sample=True, top_p=0.8, temperature=tem, max_new_tokens=128)[:, num_request_tokens:]
                            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        elif model_name == "Yi":
                            messages = [
                            {
                                "role": "user", 
                                "content": request
                            }
                            ]
                            input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
                            num_request_tokens = input_ids.shape[1] 
                            outputs = model.generate(input_ids.cuda(), eos_token_id=tokenizer.eos_token_id, num_beams=10, repetition_penalty=1.2,
                                         num_return_sequences=10, do_sample=True, top_p=0.8, temperature=tem, max_new_tokens=128)[:, num_request_tokens:]
                            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        elif model_name == "ChatGLM":
                            inputs = tokenizer.build_chat_input(request, history=[], role="user")
                            num_request_tokens = inputs.input_ids.shape[1]
                            eos_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"),
                                tokenizer.get_command("<|observation|>")]
                            outputs = model.generate(inputs.input_ids.cuda(), eos_token_id=eos_token_id, num_beams=10, 
                                         num_return_sequences=10, do_sample=True, top_p=0.8, temperature=tem, max_new_tokens=128)[:, num_request_tokens:]
                            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        elif model_name == "BaiChuan":
                            messages = [
                                {
                                "role": "user", 
                                "content": request
                                }
                            ]
                            input_tokens = [model.generation_config.user_token_id] \
                                            + tokenizer.encode(messages[0]["content"]) \
                                            + [model.generation_config.assistant_token_id]
                            input_ids = torch.LongTensor([input_tokens])
                            num_request_tokens = input_ids.shape[1]
                            outputs = model.generate(input_ids.cuda(), eos_token_id=tokenizer.eos_token_id, num_beams=10, repetition_penalty=1.2,
                                         num_return_sequences=10, do_sample=True, top_p=0.85, temperature=tem, max_new_tokens=128)[:, num_request_tokens:]
                            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        elif model_name == "Alpaca":
                            messages = [
                                {
                                    "role": "system", 
                                    "content": "You are a helpful and honest assistant."
                                },
                                {
                                    "role": "user", 
                                    "content": request,
                                },
                            ]
                            input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
                            num_request_tokens = input_ids.shape[1]
                            outputs = model.generate(input_ids.cuda(), eos_token_id=tokenizer.eos_token_id, num_beams=10, 
                                         num_return_sequences=10, do_sample=True, top_p=0.95, temperature=tem, max_new_tokens=128)[:, num_request_tokens:]
                            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        output_list.append(
                            {
                                "request": request,
                                "response":keep_last_sentence(random.choice(outputs)),
                            }
                        )
                        break
                    except:
                        tem -= 0.1
                        tem = max(tem, 1.0)
            response[type_].append(output_list)
        folder_name = f"./outputs/cn/loop-{loop_num}/responses/{model_name}/{type_}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        with open(os.path.join(folder_name, "baseline.json"), "w", encoding="utf-8") as f:
            json.dump(response[type_], f, ensure_ascii=False, indent=4)



def generate_answer_BS(all_data, model_name, model, tokenizer, loop_num):
    response = defaultdict(list)
    for type_, data in all_data.items():
        folder_name = f"./outputs/cn/loop-{loop_num}/responses/{model_name}/{type_}"
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
            for request in [item["request"] for item in items]:
                tem = 1.2
                num_beams = 10
                method = "fast" # 快，但是有小概率失效
                if items[0]["category"] in ["创意写作", "迂回描述"]:
                    constraints = [items[0]["constraints"]+[" ", "-"]+others]
                else:
                    constraints = [items[0]["constraints"]+[" ", "-"]+others+degrees+commands+sensitives]
                while True:
                    try:
                        if model_name == "Deepseek":
                            outputs = BeamSearchWarpperDeepseek(request, model, tokenizer, num_beams=num_beams, num_returns=10, 
                                    bad_words_list=constraints, do_sample=True, top_p=0.95, 
                                    temperature=tem, max_length=128, method=method,
                                    early_stopping=True, use_cache=True)[0]
                        elif model_name == "Qwen":
                            outputs = BeamSearchWarpperQwen(request, model, tokenizer, num_beams=num_beams, num_returns=10, 
                                    bad_words_list=constraints, do_sample=True, top_p=0.8, 
                                    temperature=tem, max_length=128, method=method,repetition_penalty=1.2,
                                   early_stopping=True, use_cache=True)[0]
                        elif model_name == "Yi":
                            outputs = BeamSearchWarpperYi(request, model, tokenizer, num_beams=num_beams, num_returns=10, 
                                    bad_words_list=constraints, do_sample=True, top_p=0.8, 
                                    temperature=tem, max_length=128, method=method, repetition_penalty=1.2,
                                    early_stopping=True, use_cache=True)[0]
                        elif model_name == "ChatGLM":
                            outputs = BeamSearchWarpperChatGLM(request, model, tokenizer, num_beams=num_beams, num_returns=10, 
                                    bad_words_list=constraints, do_sample=True, top_p=0.8, 
                                    temperature=tem, max_length=128, method=method,
                                    early_stopping=True, use_cache=False)[0]
                        elif model_name == "BaiChuan":
                            outputs = BeamSearchWarpperBaiChuan(request, model, tokenizer, num_beams=num_beams, num_returns=10, 
                                    bad_words_list=constraints, do_sample=True, top_p=0.85, 
                                    temperature=tem, max_length=128, method=method,repetition_penalty=1.2,
                                    early_stopping=True, use_cache=True)[0]
                        elif model_name == "Alpaca":
                            outputs = BeamSearchWarpperAlpaca(request, model, tokenizer, num_beams=num_beams, num_returns=10, 
                                    bad_words_list=constraints, do_sample=True, top_p=0.95, repetition_penalty=1.2,
                                    temperature=tem, max_length=128, method=method,
                                    early_stopping=True, use_cache=True)[0]
                        output_list.append(
                            {
                                "category": items[0]["category"],
                                "request": request,
                                "response":keep_last_sentence(random.choice(outputs)),
                            }
                        )
                        break
                    except:
                        method = "stable"  # 慢，但是绝对不会失效
            response[type_].append(output_list)
            folder_name = f"./outputs/cn/loop-{loop_num}/responses/{model_name}/{type_}"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            with open(os.path.join(folder_name, "BS.json"), "w", encoding="utf-8") as f:
                json.dump(response[type_], f, ensure_ascii=False, indent=4)





def get_merged_tfidf_result(loop_num=0, merged_models=None, implemention="jieba", normalization=None, PMI=None):
    assert implemention in ["jieba"] # 可以添加预训练
    assert normalization in [None, "linf", "l1", "l2"]
    assert PMI in [None, "lp", "prob"]
    directory = os.path.join(f'./outputs/cn', f"loop-{loop_num}", "responses")
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
    scores = get_merged_tfidf_result_jieba(all_direct_items, all_indirect_items, normalization, PMI)
    if PMI is None:
        path = f"./outputs/cn/loop-{loop_num}/constraints_select/jieba/rank"
    else:
        path = f"./outputs/cn/loop-{loop_num}/constraints_select/jieba/PMI/{PMI}/{normalization}" if normalization else  f"./outputs/cn/loop-{loop_num}/constraints_select/jieba/PMI/{PMI}"
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


def PMI_score(tfidf_dict, PMI):
    """
    tfidf : {word:tfidf}
    """
    words = list(tfidf_dict.keys())
    tfidf = np.array(list(tfidf_dict.values()))
    if PMI == "lp":
        tfidf = np.log(tfidf)
    elif PMI == "prob":
        tfidf = np.log(1-np.exp(-tfidf))
    return dict(zip(words, tfidf))

def compute_PMI_difference(all_direct_tfidf, all_indirect_tfidf, PMI, default_value="min"):
    """
    PMI: lp, prob
    """
    computed_PMI = []
    for direct_tfidf, indirect_tfidf in zip(all_direct_tfidf, all_indirect_tfidf):
        direct_PMI = PMI_score(direct_tfidf, PMI)
        indirect_PMI = PMI_score(indirect_tfidf, PMI)
        computed_PMI_i = {}
        if default_value == "min":
            direct_default = min(direct_PMI.values())
            indirect_default = min(indirect_PMI.values())
        elif default_value == "median":
            direct_default = statistics.median(list(direct_PMI.values()))
            indirect_default = statistics.median(list(indirect_PMI.values()))
        words = set().union(list(direct_PMI.keys()), list(indirect_PMI.keys()))
        for word in words:
            direct_score = direct_PMI.get(word, direct_default)
            indirect_score = indirect_PMI.get(word, indirect_default)
            score = direct_score - indirect_score
            computed_PMI_i[word] = score
        computed_PMI_i = dict(sorted(computed_PMI_i.items(), key=lambda x:x[1], reverse=True))
        computed_PMI_i_in_indirect = {word:score for word, score in computed_PMI_i.items() if word in indirect_tfidf.keys()}
        computed_PMI.append((computed_PMI_i, computed_PMI_i_in_indirect))
    return computed_PMI


    
def get_merged_tfidf_result_jieba(all_direct_items, all_indirect_items, normalization=None, PMI=None):
    all_direct_text = {}
    all_indirect_text = {}
    for i, items in all_direct_items.items():
        all_direct_text[i] = "".join([item["response"] for item in items])
    for i, items in all_indirect_items.items():
        all_indirect_text[i] = "".join([item["response"] for item in items])
    # 计算direct和indirect的tfidf
    all_direct_tfidf = []
    all_indirect_tfidf = []
    for _, text in all_direct_text.items():
        all_direct_tfidf.append(jieba_tfidf(text, idf_freq, median_idf, min_idf, normalization=normalization, PMI=PMI))
    for _, text in all_indirect_text.items():
        all_indirect_tfidf.append(jieba_tfidf(text, idf_freq, median_idf, min_idf, normalization=normalization, PMI=PMI))
    if PMI is None:
        # 排序差方法
        compute_rank = compute_rank_difference(all_direct_tfidf, all_indirect_tfidf)
        return compute_rank
    else:
        # PMI方法
        compute_PMI_score = compute_PMI_difference(all_direct_tfidf, all_indirect_tfidf, PMI)
        return compute_PMI_score
    
def logsigmoid(x):
    return -np.log((1+np.exp(-x)))

def add_sentiment_score(file, out_file, max_alpha=1):
    with open(file, "r", encoding="utf-8") as f:
        tfidf_scores = json.load(f)
    new_tfidf_scores = []
    for i, (direct_scores, indirect_scores) in tqdm(enumerate(tfidf_scores)):
        if i > 58:
            alpha = max_alpha
        else:
            alpha = 0
        words = list(direct_scores.keys())
        for word in words:
            score = np.clip((1+words_sentiment_scores.get(word, 0))/2, 1e-3, 1-1e-3) # P(POS)
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
    with open(f"./outputs/cn/loop-{loop}/requests/{model_name}/data_direct.json", "r", encoding="utf-8") as f:
        data_direct = json.load(f)
    with open(f"./outputs/cn/loop-{loop}/requests/{model_name}/data_indirect.json", "r", encoding="utf-8") as f:
        data_indirect = json.load(f)
    new_data_direct = []
    new_data_indirect = []
    for direct_items, indirect_items, item_words in zip(data_direct, data_indirect, words):
        new_item_data_direct = []
        new_item_data_indirect = []
        new_add_words = list(set(list(item_words[0].keys())[:topK]+list(item_words[1].keys())[:topK]))
        for item in direct_items:
            item["constraints"] += new_add_words
            item["constraints"] = list(set(item["constraints"]))
            if "还要注意，你被要求不能使用以下词语：" not in item["request"]:
                item["request"] += "还要注意，你被要求不能使用以下词语：" + "，".join(item["constraints"]) + "。"
            else:
                item["request"] = item["request"].split("还要注意，你被要求不能使用以下词语")[0]
                item["request"] += "还要注意，你被要求不能使用以下词语：" + "，".join(item["constraints"]) + "。"
            new_item_data_direct.append(item)
        for item in indirect_items:
            item["constraints"] += new_add_words
            item["constraints"] = list(set(item["constraints"]))
            if "还要注意，你被要求不能使用以下词语：" not in item["request"]:
                item["request"] += "还要注意，你被要求不能使用以下词语：" + "，".join(item["constraints"]) + "。"
            else:
                item["request"] = item["request"].split("还要注意，你被要求不能使用以下词语")[0]
                item["request"] += "还要注意，你被要求不能使用以下词语：" + "，".join(item["constraints"]) + "。"
            new_item_data_indirect.append(item)
        new_data_direct.append(new_item_data_direct)
        new_data_indirect.append(new_item_data_indirect)
    path = f"./outputs/cn/loop-{loop+1}/requests/{model_name}"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, "data_direct.json"), "w", encoding="utf-8") as f:
       json.dump(new_data_direct, f, indent=4, ensure_ascii=False)
    with open(os.path.join(path, "data_indirect.json"), "w", encoding="utf-8") as f:
       json.dump(new_data_indirect, f, indent=4, ensure_ascii=False)
        

    