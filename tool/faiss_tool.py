# -*- coding:utf-8 -*-
'''
# @FileName    :faiss_tool.py
---------------------------------
# @Time        :2023/9/8 
# @Author      :zhangyj-n
# @Email       :13091375161@163.com
# @zhihu       :https://www.zhihu.com/people/zhangyj-n
# @kg-nlp      :https://kg-nlp.github.io/Algorithm-Project-Manual/
---------------------------------
# 目标任务 :  使用faiss构建向量库
---------------------------------
'''

import faiss
import numpy as np
import pandas as pd
from loguru import logger
import json
from tqdm import tqdm

class Faiss_recall():
    def __init__(self,**kwargs):
        # Initializing index

        dim = kwargs['output_embedding_size']
        self.space = kwargs['distance']  # metric
        nlist  = kwargs['nlist']  # 聚类中心个数
        if self.space == 'ip':
            self.index = faiss.IndexFlatIP(dim)
        elif self.space == 'l2':
            self.index = faiss.IndexFlatL2(dim)

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
    def faiss_index(self,npy_file):
        logger.info('loading... '+npy_file)
        all_embeddings = np.load(npy_file)
        data_size = all_embeddings.shape[0]
        batch_size = 10000
        for i in tqdm(range(0, data_size, batch_size)):
            cur_end = i + batch_size
            if cur_end > data_size:
                cur_end = data_size
            batch_emb = all_embeddings[np.arange(i, cur_end)]
            self.index.add(batch_emb.astype('float32'))
            # faiss.write_index(indexer, collection_name)
        logger.info("index类说明 "+str(self.index))
        return self.index

    '''向量召回,输入query_embedding,输出结果文件'''
    def faiss_recall_file(self,query_total_embedding:list,query_list:list,pre_batch_size,recall_result_file:str,recall_num):

        total_list = []
        with open(recall_result_file, "w", encoding="utf-8") as f:
            if self.space == 'ip':
                for batch_index, batch_query_embedding in enumerate(query_total_embedding):
                    cosine_sims,recalled_idx = self.index.search(batch_query_embedding, recall_num)
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
                    l2_sims,recalled_idx  = self.index.search(batch_query_embedding, recall_num)
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
            cosine_sims,recalled_idx = self.index.search(batch_query_embedding, recall_num)
            batch_size = len(cosine_sims)
            for row_index in range(batch_size):
                for idx, doc_idx in enumerate(recalled_idx[row_index]):
                    total_list.append([query_list[row_index], self.id2corpus[doc_idx], 1-cosine_sims[row_index][idx]])
        elif self.space == 'l2':
            for batch_index, batch_query_embedding in enumerate(query_total_embedding):
                l2_sims,recalled_idx = self.index.search(batch_query_embedding, recall_num)
                batch_size = len(l2_sims)
                for row_index in range(batch_size):
                    for idx, doc_idx in enumerate(recalled_idx[row_index]):
                        total_list.append([query_list[row_index], self.id2corpus[doc_idx], l2_sims[row_index][idx]])
        print(total_list)


if __name__ == '__main__':
    pass

