import jieba
import numpy as np
import jieba.analyse
import jieba.posseg
from operator import itemgetter

global stop_words
global allowPOS
global idf_freq

with open('datasets/cn/hit_stopwords.txt', 'r') as f:
    stop_words = set(line.strip() for line in f)
stop_words.add(' ')
stop_words.add('\n')
allowPOS = ['a', 'n', 'v', 'd']

jieba.analyse.set_stop_words('datasets/cn/hit_stopwords.txt')

def jieba_tfidf(text, idf_freq, median_idf, min_idf, topK=None, normalization=None, PMI=None):
    if PMI !=  "prob":
        word_tfidf = jieba.analyse.extract_tags(text, withWeight=True, allowPOS=allowPOS, topK=20)
    else:
        word_tfidf = extract_tags_complex(text, idf_freq, median_idf, min_idf, withWeight=True, allowPOS=allowPOS, PMI=PMI)
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

    
def extract_tags_complex(sentence, idf_freq, median_idf, min_idf, topK=20, withWeight=False, allowPOS=(), withFlag=False, PMI=None):
    tokenizer = jieba.dt
    postokenizer = jieba.posseg.dt
    if allowPOS:
        allowPOS = frozenset(allowPOS)
        words = postokenizer.cut(sentence)
    else:
        words = tokenizer.cut(sentence)
    freq = {}
    for w in words:
        if allowPOS:
            if w.flag not in allowPOS:
                continue
            elif not withFlag:
                w = w.word
        wc = w.word if allowPOS and withFlag else w
        if len(wc.strip()) < 2 or wc.lower() in stop_words:
            continue
        try:
            freq[w] += 1
        except:
            freq[w] = 1
    for k in freq:
        kw = k.word if allowPOS and withFlag else k
        freq[k] = freq[k]*idf_freq.get(kw, median_idf)        
    total = sum(freq.values())
    freq = {x:y/total for x, y in freq.items()}
    if withWeight:
        tags = sorted(freq.items(), key=itemgetter(1), reverse=True)
    else:
        tags = sorted(freq, key=freq.__getitem__, reverse=True)
    if topK:
        return tags[:topK]
    else:
        return tags


