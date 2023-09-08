# -*- coding:utf-8 -*-
'''
# @FileName    :recall.py
---------------------------------
# @Time        :2023/9/6 
# @Author      :zhangyj-n
# @Email       :13091375161@163.com
# @zhihu       :https://www.zhihu.com/people/zhangyj-n
# @kg-nlp      :https://kg-nlp.github.io/Algorithm-Project-Manual/
---------------------------------
# 目标任务 :  召回
---------------------------------
'''

from tool.vector_extract import get_single_vector
from tool.data import gen_text_file
from tqdm import tqdm
import os
import pandas as pd
import multiprocessing
import numpy as np
from loguru import logger

current_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(current_dir,'data')
result_dir = os.path.join(current_dir,'recall_result_dir')

model_dim = {
    'm3e-small':512,
    'm3e-base':768,
    'm3e-large':1024 ,
    'bge-small-zh':512 ,
    'bge-base-zh':768 ,
    'bge-large-zh':1024,
    'text2vec-base-chinese':768,
    'text2vec-large-chinese':1024,
    'text2vec-base-chinese-sentence':768,
    'text2vec-base-chinese-paraphrase':768,
    'text2vec-bge-large-chinese':1024,
    'Ernie-model/ernie-1.0-base-zh':768,
    'Ernie-model/ernie-1.0-large-zh-cw':1024,
    "Ernie-model/ernie-3.0-medium-zh":768,
    'Ernie-model/ernie-3.0-base-zh':768,
    'Ernie-model/rocketqa-zh-dureader-query-encoder':768,
    'Ernie-model/rocketqa-zh-base-query-encoder': 768
}

'''根据验证集,召回库,输出结果'''
def get_dev_result(model_name,corpus_file,npy_file,query_file,recall_result_file,tool_type,model_type,batch_size):
    '''
    Args:
        model_name: 向量模型名称
        corpus_file: # 召回库语料
        npy_file: # 经过vector_extract抽取后的numpy向量文件
        query_file: # 验证集
        recall_result_file: 召回的结果
        tool_type: 选择向量工具["hnswlib","milvus","faiss"]
    Returns:
    '''
    # 读取验证集
    logger.info('读取验证集')
    query_list,_ = gen_text_file(query_file)

    # 提取验证集向量
    if model_type == 'torch':
        if model_name.startswith('bge'):
            prompt_query_list= ['为这个句子生成表示以用于检索相关文章：'+i for i in query_list]
            query_total_embedding = get_single_vector(model_name=model_name,corpus_list=prompt_query_list,batch_size=batch_size,model_type=model_type)
        else:
            query_total_embedding = get_single_vector(model_name=model_name,corpus_list=query_list,batch_size=batch_size,model_type=model_type)
    elif model_type == 'paddle':
        query_total_embedding = get_single_vector(model_name=model_name,corpus_list=query_list,batch_size=batch_size,model_type=model_type)
    # 实例化向量工具
    if tool_type == 'hnswlib':
        from tool.hnswlib_tool import Hnswlib_recall
        parameter = {
            "output_embedding_size":model_dim[model_name],
            "distance":"ip",
            "hnsw_max_elements":1000000,
            "hnsw_ef":100,
            "hnsw_m":100,
            "hnsw_threads":4,
        }
        hnswlib_recall = Hnswlib_recall(**parameter)
        # 获取召回库向量id映射
        hnswlib_recall.gen_id2corpus(corpus_file)
        # 向量存储
        hnswlib_recall.hnswlib_index(npy_file)
        # 召回结果文件
        hnswlib_recall.hnswlib_recall_file(query_total_embedding=query_total_embedding,query_list=query_list,recall_result_file=recall_result_file,pre_batch_size=batch_size,recall_num=10)
    elif tool_type == 'faiss':
        from tool.faiss_tool import Faiss_recall
        parameter = {
            "output_embedding_size": model_dim[model_name],
            "distance": "ip",
            "nlist": 10,
        }
        faiss_recall = Faiss_recall(**parameter)
        # 获取召回库向量id映射
        faiss_recall.gen_id2corpus(corpus_file)
        # 向量存储
        faiss_recall.faiss_index(npy_file)
        # 召回结果文件
        faiss_recall.faiss_recall_file(query_total_embedding=query_total_embedding, query_list=query_list,
                                           recall_result_file=recall_result_file, pre_batch_size=batch_size,
                                           recall_num=10)
    elif tool_type == 'milvus':
        from tool.milvus_tool import Milvus_recall
        from milvus import MetricType, IndexType
        if 'Ernie-model' in model_name:
            collection_name = model_name.replace('Ernie-model/','').replace('-',"_").replace('.','_')
        else:
            collection_name = model_name.replace('-',"_").replace('.','_')
        parameter = {
            "collection_name":collection_name,
            "partition_tag": "partition_tag_1",
            "index_type": IndexType.FLAT,
            "nlist":1000,
            "nprobe":100,
            "dimension":model_dim[model_name],
            "metric_type":MetricType.IP,
            "index_file_size":256,
        }
        milvus_recall = Milvus_recall(**parameter)
        # 获取召回库向量id映射
        milvus_recall.gen_id2corpus(corpus_file)
        # 向量存储
        milvus_recall.milvus_index(npy_file,cover=False)
        # 召回结果文件
        milvus_recall.milvus_recall_file(query_total_embedding=query_total_embedding, query_list=query_list,
                                       recall_result_file=recall_result_file, pre_batch_size=batch_size,
                                       recall_num=10)

if __name__ == '__main__':
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

    corpus_file = os.path.join(data_dir,'corpus.tsv')
    npy_file = os.path.join(data_dir,model_name+'.npy')
    query_file = os.path.join(data_dir,'dev.tsv')
    recall_result_file = os.path.join(result_dir,'recall_result.txt')
    get_dev_result(model_name,corpus_file,npy_file,query_file,recall_result_file,tool_type='hnswlib',model_type='paddle',batch_size=128)

    pass


