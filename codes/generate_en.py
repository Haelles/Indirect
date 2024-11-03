import json
import os
import torch
from utils import *
from GenerateResponsesEN import * 
from ConstrainedBeamSearch import *

models = ["Mistral"]

generate_direct_or_indirect_requests(num=5, model_names=models)

for loop in [1,2,3]:
    for model_name in models:
        all_data = load_direct_indirect(loop, model_name)
        model, tokenizer = load_model(model_name)
        generate_answer_baseline(all_data, model_name, model, tokenizer, loop_num=loop)
        generate_answer_BS(all_data, model_name, model, tokenizer, loop_num=loop)
        model.cpu()
        torch.cuda.empty_cache()
        del model
        del tokenizer

        get_merged_tfidf_result(loop_num=loop, normalization="l1", PMI="lp", merged_models=[model_name])
        max_alpha = 1.0
        add_sentiment_score(f"./outputs/en/loop-{loop}/constraints_select/pretrained/PMI/lp/l1/{model_name}.json", f"./outputs/en/loop-{loop}/constraints_select-sentiment/pretrained/PMI/lp/l1/{model_name}", max_alpha=max_alpha)
        add_constraints(loop, f"./outputs/en/loop-{loop}/constraints_select-sentiment/pretrained/PMI/lp/l1/{model_name}/{max_alpha}.json", model_name, topK=5)