# -*- coding:utf-8 -*-
'''
# @FileName    :evaluate.py
---------------------------------
# @Time        :2023/9/6 
# @Author      :zhangyj-n
# @Email       :zhangyj-n@glodon.com
---------------------------------
# 目标任务 :    评估向量召回率
---------------------------------
'''
import argparse
import os
import numpy as np


current_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(current_dir,'data')
result_dir = os.path.join(current_dir,'recall_result_dir')

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--similar_text_pair", type=str, default=os.path.join(data_dir,'dev.tsv'), help="The full path of similat pair file")
parser.add_argument("--recall_result_dir", type=str, default='./recall_result_dir', help="The full path of recall result file to save")
parser.add_argument("--recall_result_file", type=str, default='recall_result.txt', help="The full path of recall result file")
parser.add_argument("--recall_topn_result", type=str, default='reslut.tsv', help="The full path of recall result")
parser.add_argument("--recall_num", type=int, default=10, help="Most similair number of doc recalled from corpus per query")
args = parser.parse_args()
# yapf: enable


def recall(rs, N=10):
    """
    Ratio of recalled Ground Truth at topN Recalled Docs
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> recall(rs, N=1)
    0.333333
    >>> recall(rs, N=2)
    >>> 0.6666667
    >>> recall(rs, N=3)
    >>> 1.0
    Args:
        rs: Iterator of recalled flag()
    Returns:
        Recall@N
    """

    recall_flags = [np.sum(r[0:N]) for r in rs]
    return np.mean(recall_flags)


def evaluate_result(model_name):
    text2similar = {}
    with open(args.similar_text_pair, "r", encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            text, similar_text = line.rstrip().split("\t")
            text2similar[text] = similar_text

    rs = []
    recall_result_file = os.path.join(args.recall_result_dir, args.recall_result_file)
    with open(recall_result_file, "r", encoding="utf-8") as f:
        relevance_labels = []
        for index, line in enumerate(f):

            if index % args.recall_num == 0 and index != 0:
                rs.append(relevance_labels)
                relevance_labels = []

            text, recalled_text, cosine_sim = line.rstrip().split("\t")
            if text2similar[text] == recalled_text:
                relevance_labels.append(1)
            else:
                relevance_labels.append(0)

    recall_N = []
    recall_num = [1, 5, 10]
    recall_topn_result = os.path.join(args.recall_result_dir, args.recall_topn_result)
    result = open(recall_topn_result, "a")
    res = []
    for topN in recall_num:
        R = round(100 * recall(rs, N=topN), 3)
        recall_N.append(str(R))
    for key, val in zip(recall_num, recall_N):
        print("recall@{}={}".format(key, val))
        res.append(str(val))
    result.write('|-|{}|{}|{}|{}| | |基础模型|'.format(model_name,recall_N[0], recall_N[1], recall_N[2]) + "\n")
    print('|-|{}|{}|{}|{}|基础模型|'.format(model_name,recall_N[0], recall_N[1], recall_N[2]))

if __name__ == "__main__":


    # model_dim = {
    #     'm3e-small':512,
    #     'm3e-base':768,
    #     'm3e-large':1024 ,
    #     'bge-small-zh':512 ,
    #     'bge-base-zh':768 ,
    #     'bge-large-zh':1024
    # }
    # model_name = 'm3e-small'  # 512
    # model_name = 'm3e-base'  # 768
    # model_name = 'm3e-large'  # 1024
    # model_name = 'bge-small-zh'  # 512
    # model_name = 'bge-base-zh'  # 768
    # model_name = 'bge-large-zh'  # 1024
    # model_name = 'text2vec-base-chinese'  # 768
    # model_name = 'text2vec-large-chinese'  # 1024
    # model_name = 'text2vec-base-chinese-sentence'  # 768
    # model_name = 'text2vec-base-chinese-paraphrase'  # 768
    # model_name = 'text2vec-bge-large-chinese'  # 1024

    model_name = 'Ernie-model/ernie-1.0-base-zh'  # 768
    # model_name = 'Ernie-model/ernie-1.0-large-zh-cw'  # 1024
    # model_name = 'Ernie-model/ernie-3.0-medium-zh'  # 768
    # model_name = 'Ernie-model/ernie-3.0-base-zh'  #  768
    # model_name = 'Ernie-model/rocketqa-zh-dureader-query-encoder'  # 768
    # model_name = 'Ernie-model/rocketqa-zh-base-query-encoder'  # 768
    evaluate_result(model_name)