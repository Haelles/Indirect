# Do Not Say It Directly: Generating Indirect Expression with Large Language Models
Code, data and results for the paper "Do Not Say It Directly: Generating Indirect Expression with Large Language Models".

## Framework
![framework](figure/framework.png)


## Requirements
Please place the model weights in the `huggingface_model/` directory, then install the conda environment.
```
conda env create -f environment.yml
conda activate indirect
```

## Data
### Chinese
The path to the original Chinese data is `datasets/cn/Dataset.json`. During the experiment, we fill the Chinese data into the template (`datasets/cn/templates.json`) as the final input, which can provide richer guidance information to the LLM.

Additionally, with each iteration, the list of constrained words will change. The path to the prompt input for the LLM in each iteration is as follows:
* loop1: `datasets/cn/loop-1`
* loop2: `datasets/cn/loop-2`
* loop3: `datasets/cn/loop-3`

In each iteration, we will update the list of constrained words based on the direct word mining algorithm. However, there are also some very common words that contain sensitive content or may harm others, which should not be used in the indirect expression. We have predefined these words and stored them at the following path:
* `datasets/cn/敏感词表.json`
* `datasets/cn/指令词.txt`


### English
The path to the original English data is `datasets/en/Dataset.json`. The path to the prompt input for the LLM in each iteration is as follows:
* loop1: `datasets/en/loop-1`
* loop2: `datasets/en/loop-2`
* loop3: `datasets/en/loop-3`

Similar to the Chinese dataset, we have also predefined a set of common words that are not suitable for use in indirect expressions.
* `datasets/en/sensitives.json`

## Code
The code in the `code` directory serves the following purposes:
* `codes/ConstrainedBeamSearch.py`: Implemented constrained beam search.
* `codes/generate_cn.py`: The entry script for the Chinese experiment.
* `codes/generate_en.py`: The entry script for the English experiment.
* `codes/GenerateResponsesCN.py`: Implemented the pipeline for Chinese experiments, including loading models, generating responses, extracting constrained words, etc.
* `codes/GenerateResponsesEN.py`: Implemented the pipeline for English experiments, including loading models, generating responses, extracting constrained words, etc.
* `codes/tfidf_cn.py`: Specifically implemented several constrained word mining algorithms involved in the Chinese experiments.
* `codes/tfidf_en.py`: Specifically implemented several constrained word mining algorithms involved in the English experiments.
* `codes/utils.py`: Helper functions.


## Experiment
### Chinese
Run `codes/generate_cn.py`

### English
Run `codes/generate_en.py`


## Results
All experimental results are stored in `paper_output/data`.


### Direct Word Mining Algorithm
HIT/F1/NDCG@k results on Chinese data is as follows.
![mining_cn](paper_output/data/cn/mining_cn.png)


HIT/F1/NDCG@k results on English data is as follows.
![mining_en](paper_output/data/en/mining_en.png)

### The Evolution of Direct Words
In each iteration, the constrained words, also known as direct words, will gradually change. For example, in Chinese, the "constraints" field of each data point in `datasets/cn/loop-2/data_indirect.json` records the constrained words for that data in loop 2, and the "constraints" field of each data point in `datasets/cn/loop-3/data_indirect.json` records the constrained words for that data in loop 3. As the iterations progress, the list of constrained words will grow longer and more challenging, which will require the LLM to generate increasingly indirect responses.

Note that the manually annotated constrained words are only used to adjust the parameters of our mining algorithm. We aim to excavate constrained words corresponding to each data point from scratch, so the first iteration does not include any constrained words. Therefore, the "constraints" field in `loop-1` is left empty.