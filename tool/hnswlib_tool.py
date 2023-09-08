# -*- coding:utf-8 -*-
'''
# @FileName    :hnswlib_tool.py
---------------------------------
# @Time        :2023/9/6 
# @Author      :zhangyj-n
# @Email       :zhangyj-n@glodon.com
---------------------------------
# 目标任务 :   使用hnswlib构建向量库
---------------------------------
'''
import hnswlib
import numpy as np
import pandas as pd
from loguru import logger
import json
class Hnswlib_recall():
    def __init__(self,**kwargs):
        # Initializing index
        # max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded
        # during insertion of an element.
        # The capacity can be increased by saving/loading the index, see below.

        # ef_construction - controls index search speed/build speed tradeoff

        # M - is tightly connected with internal dimensionality of the data. Strongly affects memory consumption (~M)
        # Higher M leads to higher accuracy/run_time at fixed ef/efConstruction

        # Controlling the recall by setting ef:
        # higher ef leads to better accuracy, but slower search

        # Set number of threads used during batch search/construction
        # By default using all available cores

        dim = kwargs['output_embedding_size']  #
        self.space = kwargs['distance']  # metric
        max_elements = kwargs['hnsw_max_elements']  # the maximum number of elements that can be stored in the structure(can be increased/shrunk)
        ef_construction = kwargs['hnsw_ef']  # a construction time/accuracy trade-off
        M = kwargs['hnsw_m']  # ha maximum number of outgoing connections in the graph
        num_threads = kwargs['hnsw_threads']  # he number of cpu threads to use
        self.index = hnswlib.Index(space=self.space, dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
        self.index.set_ef(ef_construction)
        self.index.set_num_threads(num_threads)
        logger.info("初始化index完成")

    '''向量id映射'''
    def gen_id2corpus(self,corpus_file):
        self.id2corpus = {}
        with open(corpus_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f.readlines()[1:]):
                self.id2corpus[idx] = line.rstrip()
        logger.info("召回库总数量:{}".format(len(self.id2corpus)))
        # print(json.dumps(self.id2corpus,indent=2,ensure_ascii=False))
        return self.id2corpus

    '''向量存储'''
    def hnswlib_index(self,npy_file):
        logger.info('loading... '+npy_file)
        all_embeddings = np.load(npy_file)
        self.index.add_items(all_embeddings)
        logger.info("总索引数:{}".format(self.index.get_current_count()))
        logger.info("index类说明 "+str(self.index))
        # print(self.index.get_ids_list())
        return self.index

    '''向量召回,输入query_embedding,输出结果文件'''
    def hnswlib_recall_file(self,query_total_embedding:list,query_list:list,pre_batch_size,recall_result_file:str,recall_num):

        total_list = []
        with open(recall_result_file, "w", encoding="utf-8") as f:
            if self.space == 'ip':
                for batch_index, batch_query_embedding in enumerate(query_total_embedding):
                    recalled_idx, cosine_sims = self.index.knn_query(batch_query_embedding, recall_num)
                    # print(recalled_idx,cosine_sims)
                    # print(recalled_idx)
                    batch_size = len(cosine_sims)

                    for row_index in range(batch_size):
                        text_index = pre_batch_size * batch_index + row_index
                        for idx, doc_idx in enumerate(recalled_idx[row_index]):
                            # total_list.append([query_list[text_index], id2corpus[doc_idx], 1.0 - cosine_sims[row_index][idx]])
                            f.write(
                                "{}\t{}\t{}\n".format(
                                    query_list[text_index], self.id2corpus[doc_idx], 1.0-cosine_sims[row_index][idx]
                                )
                            )
            elif self.space == 'l2':
                for batch_index, batch_query_embedding in enumerate(query_total_embedding):
                    recalled_idx, l2_sims = self.index.knn_query(batch_query_embedding, recall_num)
                    batch_size = len(l2_sims)
                    for row_index in range(batch_size):
                        text_index = pre_batch_size * batch_index + row_index
                        for idx, doc_idx in enumerate(recalled_idx[row_index]):
                            f.write(
                                "{}\t{}\t{}\n".format(
                                    query_list[text_index], self.id2corpus[doc_idx], l2_sims[row_index][idx]
                                )
                            )

    '''向量召回,输入query_embedding,输出单个或几个召回结果'''
    def hnswlib_search(self,batch_query_embedding:list,query_list:list,recall_num):
        total_list = []
        if self.space == 'ip':
            recalled_idx, cosine_sims = self.index.knn_query(batch_query_embedding, recall_num)
            batch_size = len(cosine_sims)
            for row_index in range(batch_size):
                for idx, doc_idx in enumerate(recalled_idx[row_index]):
                    total_list.append([query_list[row_index], self.id2corpus[doc_idx], 1-cosine_sims[row_index][idx]])
        elif self.space == 'l2':
            for batch_index, batch_query_embedding in enumerate(query_total_embedding):
                recalled_idx, l2_sims = self.index.knn_query(batch_query_embedding, recall_num)
                batch_size = len(l2_sims)
                for row_index in range(batch_size):
                    for idx, doc_idx in enumerate(recalled_idx[row_index]):
                        total_list.append([query_list[row_index], self.id2corpus[doc_idx], l2_sims[row_index][idx]])
        print(total_list)
if __name__ == '__main__':

    pass


