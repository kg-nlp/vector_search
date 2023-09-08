# -*- coding:utf-8 -*-
'''
# @FileName    :milvus_tool.py
---------------------------------
# @Time        :2023/9/8 
# @Author      :zhangyj-n
# @Email       :13091375161@163.com
# @zhihu       :https://www.zhihu.com/people/zhangyj-n
# @kg-nlp      :https://kg-nlp.github.io/Algorithm-Project-Manual/
---------------------------------
# 目标任务 :  使用milvus构建向量库
---------------------------------
'''
from conf.config import MILVUS_HOST,MILVUS_PORT
from milvus import Milvus
import numpy as np
import pandas as pd
from loguru import logger
import json
from tqdm import tqdm


'''milvus配置'''
class VecToMilvus:
    def __init__(self):
        self.client = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)

    def has_collection(self, collection_name):
        try:
            status, ok = self.client.has_collection(collection_name)
            return ok
        except Exception as e:
            logger.info("Milvus 读取 {} 错误:".format(collection_name)+str(e))

    def creat_collection(self, collection_name,collection_param):
        try:
            status = self.client.create_collection(collection_param)
            return status
        except Exception as e:
            logger.info("Milvus 创建集合 {} 错误:".format(collection_name)+str(e))

    def create_index(self, collection_name,index_type, index_param):
        try:
            status = self.client.create_index(collection_name, index_type, index_param)
            return status
        except Exception as e:
            logger.info("Milvus 创建索引 {} 错误:".format(collection_name)+str(e))

    def has_partition(self, collection_name, partition_tag):
        try:
            status, ok = self.client.has_partition(collection_name, partition_tag)
            return ok
        except Exception as e:
            logger.info("Milvus 分区 {} 错误: ".format(partition_tag)+str(e))

    def delete_partition(self, collection_name, partition_tag):
        try:
            status = self.client.drop_partition(collection_name,partition_tag)
            return status
        except Exception as e:
            logger.info("Milvus 删除分区 {} error: ".format(partition_tag)+str(e))

    def create_partition(self, collection_name, partition_tag):
        try:
            status = self.client.create_partition(collection_name, partition_tag)
            return status
        except Exception as e:
            logger.info("Milvus 创建分区 {} error: ".format(partition_tag)+str(e))

    def insert(self, vectors, index_type, index_param,ids=None, partition_tag=None,**collection_param):

        collection_name = collection_param['collection_name']
        try:
            if not self.has_collection(collection_name):
                self.creat_collection(collection_name,collection_param)
                self.create_index(collection_name,index_type, index_param)
                logger.info("集合信息: {}".format(self.client.get_collection_info(collection_name)[1]))
            if (partition_tag is not None) and (not self.has_partition(collection_name, partition_tag)):
                self.create_partition(collection_name, partition_tag)
            status, ids = self.client.insert(
                collection_name=collection_name, records=vectors, ids=ids, partition_tag=partition_tag
            )
            self.client.flush([collection_name])
            logger.info(
                "插入 {} 数据, 还有 {} 数据.".format(
                    len(ids), self.client.count_entities(collection_name)[1]
                )
            )
            return status, ids
        except Exception as e:
            logger.info("Milvus 插入错误:"+str(e))

'''召回'''
class RecallByMilvus:
    def __init__(self):
        self.client = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)

    def search(self, collection_name,query_records, recall_num,search_param, partition_tag=None):
        try:
            status, results = self.client.search(
                collection_name=collection_name,
                query_records=query_records,
                top_k=recall_num,
                params=search_param,
                partition_tag=partition_tag,
            )
            return status, results
        except Exception as e:
            logger.info("Milvus 召回错误: "+ str(e))


class Milvus_recall():
    def __init__(self,**kwargs):
        # Initializing index
        self.collection_name = kwargs['collection_name']
        self.partition_tag = kwargs['partition_tag']
        self.index_type = kwargs['index_type']
        self.index_param = {"nlist":kwargs['nlist']}  # 每个文件中的向量类的个数，默认值为 16384
        self.search_param = {"nprobe":kwargs['nprobe']}  # 查询所涉及的向量类的个数。nprobe 影响查询精度。数值越大，精度越高，但查询速度更慢。[1,nlist]
        self.dimension = kwargs['dimension']
        self.metric_type = kwargs['metric_type']
        self.index_file_size = kwargs['index_file_size']  # 触发创建索引的阈值。该参数指定只有当原始数据文件大小达到某一阈值，系统才会为其创建索引，默认值为1024 MB。小于该阈值的数据文件不会创建索引。
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
    def milvus_index(self,npy_file,cover=False):
        logger.info('loading... '+npy_file)
        all_embeddings = np.load(npy_file)
        data_size = all_embeddings.shape[0]
        embedding_ids = [i for i in range(data_size)]
        batch_size = 10000

        client = VecToMilvus()
        if client.has_partition(self.collection_name, self.partition_tag):
            if cover:
                client.delete_partition(self.collection_name, self.partition_tag)
            else:
                logger.info('已经存在集合分区 {} {} '.format(self.collection_name,self.partition_tag))
                return

        for i in tqdm(range(0, data_size, batch_size)):
            cur_end = i + batch_size
            if cur_end > data_size:
                cur_end = data_size
            batch_emb = all_embeddings[np.arange(i, cur_end)]
            client.insert(
                vectors=batch_emb.tolist(),
                index_type=self.index_type,
                index_param=self.index_param,
                ids=embedding_ids[i: i + batch_size],
                partition_tag=self.partition_tag,
                **{"dimension":self.dimension,"index_file_size":self.index_file_size,"metric_type":self.metric_type,"collection_name":self.collection_name}
            )
        logger.info("index类说明 "+str(client))

    '''向量召回,输入query_embedding,输出结果文件'''
    def milvus_recall_file(self,query_total_embedding:list,query_list:list,pre_batch_size,recall_result_file:str,recall_num):

        client = RecallByMilvus()
        total_list = []
        with open(recall_result_file, "w", encoding="utf-8") as f:
            for batch_index, batch_query_embedding in enumerate(query_total_embedding):
                status, results = client.search(
                    collection_name=self.collection_name, query_records=batch_query_embedding,
                    recall_num=recall_num,search_param=self.search_param,partition_tag=self.partition_tag,
                )
                batch_size = len(results)
                for row_index in range(batch_size):
                    text_index = pre_batch_size * batch_index + row_index
                    for item in results[row_index]:
                        idx = item.id
                        f.write(
                            "{}\t{}\t{}\n".format(
                                query_list[text_index], self.id2corpus[idx], item.distance
                            )
                        )

if __name__ == '__main__':
    pass