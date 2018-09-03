# Joint entity recognition and relation extraction as a multi-head selection problem

Implementation of the papers
[Joint entity recognition and relation extraction as a multi-head selection problem](https://arxiv.org/abs/1804.07847) and 
[Adversarial training for multi-context joint entity and relation extraction](https://arxiv.org/abs/1808.06876).

# Requirements
* Ubuntu 16.04
* Anaconda 5.0.1
* Numpy 1.14.1
* Tensorflow 1.5.0

## Task
Given a sequence of tokens (i.e., sentence), (i) give the entity tag of each word (e.g., NER) and (ii) the relations between the entities in the sentence. The following example indicates the accepted input format of our multi-head selection model:


```
0	Marc		B-PER		['N']					[0]		
1	Smith		I-PER 		['lives_in','works_for']		[5,11]
2 	lives		O		['N']					[0]
3	in		O		['N']					[0]
4	New		B-LOC		['N']					[0]
5	Orleans		I-LOC		['N']					[0] 
6	and		O		['N']					[0]
7	is		O		['N']					[0]
8	hired		O		['N']					[0]
9	by		O		['N']					[0]
10	the		O		['N']					[0]
11  government		B-ORG		['N']					[0]
12	.		O		['N']					[0]
```

## Configuration
The model has several parameters such as: 
* EC (entity classification) or BIO (BIO encoding scheme)
* Character embeddings
* Ner loss (softmax or CRF)

that could be specified in the configuration files (see [config](https://github.com/bekou/multihead_joint_entity_relation_extraction/tree/master/configs)).

## Run the model

```
./run.sh
```

## More details
Commands executed in ```./run.sh```:

1. Train on the training set and evaluate on the dev set to obtain early stopping epoch
```python3 -u train_es.py```
2. Train on the concatenated (train + dev) set and evaluate on the test set until (1) max epochs limit exceeded or #(2) the limit specified by early stopping after executing train_es.py
```python3 -u train_eval.py```
