import torch
import torch.nn.functional as F
from utils import *
from GenerateResponsesCN import * 
from ConstrainedBeamSearch import *
from tfidf_cn import extract_tags_complex

models = ["Qwen"]

generate_direct_or_indirect_requests(num=5)

for loop in [1, 2, 3]:
    for model_name in models:
        # generate
        all_data = load_direct_indirect(loop, model_name)
        model, tokenizer = load_model(model_name)
        generate_answer_baseline(all_data, model_name, model, tokenizer, loop_num=loop)
        generate_answer_BS(all_data, model_name, model, tokenizer, loop_num=loop)
        model.cpu()
        torch.cuda.empty_cache()
        del model
        del tokenizer

        get_merged_tfidf_result(loop_num=loop, implemention="jieba", DWM="prob", merged_models=[model_name])
        max_alpha = 0.5
        add_sentiment_score(f"./outputs/cn/loop-{loop}/constraints_select/jieba/DWM/prob/{model_name}.json", f"./outputs/cn/loop-{loop}/constraints_select_sentiment/jieba/DWM/prob/{model_name}", max_alpha=max_alpha)
        add_constraints(loop, f"./outputs/cn/loop-{loop}/constraints_select_sentiment/jieba/DWM/prob/{model_name}/{max_alpha}.json", model_name, topK=5)
    