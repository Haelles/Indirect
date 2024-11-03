import re
import numpy as np
from operator import itemgetter

global stop_words
global allowPOS
global idf_freq

stop_words = []
allowPOS = ["ADJ", "ADV", "NOUN", "VERB"] 
def tokenizer(text, nlp):  
    text = " ".join(re.findall(r'\b\w+\b', text))
    tokens = nlp(text)
    tokens = [(token.lemma_.lower(), token.pos_) for token in tokens]
    return tokens

def pretrained_tfidf(text, idf_freq, median_idf, nlp, topK=None, normalization=None, PMI=None):
    word_tfidf = extract_tags_complex(text, idf_freq, median_idf, nlp, PMI=PMI)
    if topK is None:
        topK = len(word_tfidf)
    if PMI is None or PMI == "lp":
        if normalization is None and PMI is None:
            return dict(word_tfidf[:topK])
        elif normalization == "l1":
            total = sum([tfidf for _, tfidf in word_tfidf]) 
            return {word:tfidf/total for word, tfidf in word_tfidf[:topK]}
        elif normalization == "l2":
            total = np.sqrt(sum(tfidf**2 for _, tfidf in word_tfidf))
            return {word:tfidf/total for word, tfidf in word_tfidf[:topK]}
        elif normalization == "linf":
            M = max([tfidf for _, tfidf in word_tfidf[:topK]])
            return {word:tfidf/M for word, tfidf in word_tfidf[:topK]}
        else:
            raise ValueError("PMI与归一化方法不匹配！")
    elif PMI == "prob":
        return dict(word_tfidf[:topK])

    
def extract_tags_complex(sentence, idf_freq, median_idf, nlp, topK=20, PMI=None):
    words = tokenizer(sentence, nlp)
    freq = {}
    for word in words:
        if word[1] not in allowPOS or len(word[0]) < 3:
            continue
        w = word[0]
        freq[w] = freq.get(w, 0) + 1
    for k in freq:
        freq[k] = freq[k]*idf_freq.get(k, median_idf)        
    total = sum(freq.values())
    freq = {x:y/total for x, y in freq.items()}
    tags = sorted(freq.items(), key=itemgetter(1), reverse=True)
    if topK:
        return tags[:topK]
    else:
        return tags



    

