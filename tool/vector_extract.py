# -*- coding:utf-8 -*-
'''
# @FileName    :vector_extract.py
---------------------------------
# @Time        :2023/9/6 
# @Author      :zhangyj-n
# @Email       :zhangyj-n@glodon.com
---------------------------------
# 目标任务 :
---------------------------------
'''
import sys
sys.path.append('/workspace/custom_project/vector_search')

from tool.data import convert_example_test,gen_id2corpus,create_dataloader
from tqdm import tqdm
import os
import pandas as pd
import multiprocessing
import numpy as np
from loguru import logger
from functools import partial
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Pad, Tuple
from paddlenlp.datasets import MapDataset
import json

current_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = current_dir.replace('tool','data')

'''单进程提取向量'''
def get_single_vector(model_name,corpus_list,batch_size=128,output_file='',num=1,model_type='torch'):
    total_embedding = []
    if model_type == 'torch':
        from get_model import get_torch_model
        model = get_torch_model(model_name)
        length = len(corpus_list)
        for i in tqdm(range(0, length, batch_size), desc=str(num)+'提取向量'):
            start = i
            end = i + batch_size
            embedding = model.encode(corpus_list[start:end], normalize_embeddings=True)
            total_embedding.append(embedding)
    elif model_type == 'paddle':
        from get_model import get_paddle_model
        tokenizer,model = get_paddle_model(model_name)
        trans_func = partial(convert_example_test, tokenizer=tokenizer, max_seq_length=256)
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # text_input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  # text_segment
        ): [data for data in fn(samples)]
        corpus_list = [{idx: text} for idx, text in enumerate(corpus_list)]
        # print(json.dumps(corpus_list,indent=2,ensure_ascii=False))
        corpus_ds = MapDataset(corpus_list)
        corpus_data_loader = create_dataloader(
            corpus_ds, mode="predict", batch_size=batch_size, batchify_fn=batchify_fn, trans_fn=trans_func
        )
        model.eval()
        with paddle.no_grad():
            for batch_data in corpus_data_loader:
                input_ids, token_type_ids = batch_data
                input_ids = paddle.to_tensor(input_ids)
                token_type_ids = paddle.to_tensor(token_type_ids)
                sequence_output, cls_embedding = model(input_ids, token_type_ids=token_type_ids,position_ids=None, attention_mask=None)
                text_embeddings = F.normalize(cls_embedding, p=2, axis=-1)
                total_embedding.append(text_embeddings.numpy())
    elif model_type == 'simcse_paddle':
        from get_model import get_paddle_simcse_model
        tokenizer, model = get_paddle_simcse_model(model_name)
        trans_func = partial(convert_example_test, tokenizer=tokenizer, max_seq_length=256)
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # text_input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  # text_segment
        ): [data for data in fn(samples)]
        corpus_list = [{idx: text} for idx, text in enumerate(corpus_list)]
        corpus_ds = MapDataset(corpus_list)
        corpus_data_loader = create_dataloader(
            corpus_ds, mode="predict", batch_size=batch_size, batchify_fn=batchify_fn, trans_fn=trans_func
        )

        for text_embeddings in model.get_semantic_embedding(corpus_data_loader):
            total_embedding.append(text_embeddings.numpy())

    total_np_embedding = np.concatenate(total_embedding, axis=0)

    if output_file:
        np.save(output_file, total_np_embedding)  # 提取向量文件
    else:
        return total_embedding  # 直接返回向量结果




'''多进程提取向量'''
def get_vector(
        model_name='',
        corpus_file = '',
        output_file = '',
        batch_size = 128,
        thread_num=1,
        model_type='torch'
):
    # 多进程提取
    corpus_list = [i.strip() for i in open(corpus_file,'r').readlines()[1:]]
    length = len(corpus_list)
    logger.info('召回库%d条'%length)
    # num_size = length//thread_num
    # progress = []
    # num = 1
    # for i in range(0,length,num_size):
    #     start = i
    #     end = i+num_size
    #     p = multiprocessing.Process(target=get_single_vector,args=(model_name,
    #                                                                corpus_list[start:end],
    #                                                                batch_size,
    #                                                                output_file.replace(model_name,model_name+str(num)),
    #                                                                num
    #                                                                ))
    #     num += 1
    #     progress.append(p)
    #     p.start()
    # for p in progress:
    #     p.join()
    # logger.info(model_name+' 多进程提取完成')
    ## 合并文件
    # num = 1
    # vector_file_list = [model_name+str(i)+'.npy' for i in list(range(num,thread_num+num+1))]
    get_single_vector(model_name, corpus_list, batch_size,
                      output_file.replace(model_name, model_name + str(1)),
                      1,model_type)
    vector_file_list = [model_name+str(1)+'.npy']

    total_embeddings = []
    for file_name in vector_file_list:
        if file_name.startswith(model_name):
            file_path = os.path.join(data_dir,file_name)
            embeddings = np.load(file_path)
            total_embeddings.append(embeddings)
            os.remove(file_path)
    total_embeddings = np.concatenate(total_embeddings, axis=0)
    # with open('./a.text','w') as fw:
    #     for i in embeddings:
    #         fw.write(str(i.tolist()[:10])+'\n')
    np.save(output_file, total_embeddings)
    logger.info(model_name+" 数据维度{}".format(total_embeddings.shape))
    logger.info(model_name+' 合并完成')

    # embeddings = np.load(output_file)
    # (model_name,'大小',embeddings.shape)
    # print(model_name,'示例',embeddings[0:1])


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
    # model_name = 'Ernie-model/rocketqa-zh-dureader-query-encoder'  # 768
    # model_name = 'Ernie-model/ernie-3.0-medium-zh'  #  768
    # model_name = 'Ernie-model/ernie-3.0-base-zh'  #  768
    # model_name = 'Ernie-model/rocketqa-zh-base-query-encoder'  # 768

    corpus_file = os.path.join(data_dir,'corpus.tsv')
    output_file = os.path.join(data_dir,model_name+'.npy')
    get_vector(model_name,corpus_file,output_file,model_type='paddle')

    pass


