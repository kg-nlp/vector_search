# vector_search
各种向量搜索工具
负责加载模型评测

[个人github](https://github.com/kg-nlp/vector_search)

> 参考资料 [向量模型训练](https://kg-nlp.github.io/Algorithm-Project-Manual/向量表示/向量搜索工具.html)


## 开发环境

* 启动
```
docker run -it -d --name custom_env \
-p 8015:22  -p 8001:8001 -p 8002:8002 -p 8003:8003 -p 8004:8004 -p 8005:8005 \
-v /data/zhangyj/custom_project:/workspace/custom_project \
--gpus all --shm-size 64g --restart=always \
-e NVIDIA_VISIBLE_DEVICES=all \
registry.cn-beijing.aliyuncs.com/zhangyj-n/paddle_torch:base-1.13.0-2.4.2-11.7 /bin/bash
```

> 工具环境

ES环境创建
小组成员解压镜像后使用
```
docker run -d --restart=always --name llm_es -p 9200:9200 -p 9300:9300 \
-e "discovery.type=single-node" registry.cn-beijing.aliyuncs.com/sg-gie/es_hot:7.7
__
docker run -d --name es_hot -p 9200:9200 -p 9300:9300 \
-v $HOME/es_7/syn:/usr/share/elasticsearch/config/synonym \
-v $HOME/es_7/dic:/usr/share/elasticsearch/plugins/jieba/dic \
-v $HOME/es_7/stop:/usr/share/elasticsearch/config/stopwords \
-e "discovery.type=single-node" registry.cn-beijing.aliyuncs.com/sg-gie/es_hot:7.7
```
Milvus环境创建(该用GPU版本)
小组成员解压镜像后使用
```
docker run -d --name milvus_cpu_1.1.1 \
-p 19530:19530 \
-p 19121:19121 \
registry.cn-beijing.aliyuncs.com/sg-gie/milvus_cpu:1.1.1
```

```
wget http://raw.githubusercontent.com/milvus-io/milvus/v1.1.1/core/conf/demo/server_config.yaml
```
> milvusdb/milvus:1.1.1-gpu-d061621-330cc6已经设置为true
```
docker pull milvusdb/milvus:1.1.1-gpu-d061621-330cc6
docker run -d --name milvus_gpu_1.1.1 --gpus all \
-p 19530:19530 \
-p 19121:19121 \
milvusdb/milvus:1.1.1-gpu-d061621-330cc6

-v /data/zhangyj/custom_project/vector_search/conf:/var/lib/milvus/conf \

```
```
Milvus可视化
docker run -d --restart=always --name milvus-display -p 3000:80 -e API_URL=http://10.0.79.103:3000 milvusdb/milvus-em:latest
可以支持1.1.1
http://10.0.79.103:3000/
```

## 功能模块


* 先执行vector_extract.py 查看数据
* 再执行recall 提取向量文件
* 最后执行evaluate 获取评测结果

* 直接执行piplines.py 抽取向量,入库,召回,评测


## 当前已测试模型

> 参考 model.open_model.py

```python
torch_model_list = [
'm3e-small',
'm3e-large',
'm3e-base',
'bge-small-zh',
'bge-base-zh',
'bge-large-zh',
'text2vec-base-chinese',
'text2vec-large-chinese',
'text2vec-base-chinese-sentence',
'text2vec-base-chinese-paraphrase',
'text2vec-bge-large-chinese',
'Ernie-model/ernie-1.0-base-zh',
'Ernie-model/ernie-1.0-large-zh-cw',
"Ernie-model/ernie-3.0-medium-zh",
'Ernie-model/ernie-3.0-base-zh',
'Ernie-model/rocketqa-zh-dureader-query-encoder',
'Ernie-model/rocketqa-zh-base-query-encoder'
]
```

> 结果

| 策略 | 模型 | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Recall@50 | 备注 |
| --- | --- | --- | --- | --- | --- | ---| --- | 
|-|m3e-small|42.647|74.265|87.5| | |基础模型|
|-|m3e-base|42.647|78.676|88.235| | |基础模型|
|-|m3e-large|36.029|61.765|72.059| | |基础模型|
|-|bge-small-zh|36.765|65.441|74.265| | |基础模型|
|-|bge-base-zh|38.235|64.706|78.676| | |基础模型|
|-|bge-large-zh|38.971|68.382|82.353| | |基础模型|
|-|text2vec-base-chinese|40.441|72.059|80.147| | |基础模型|
|-|text2vec-large-chinese|38.235|77.941|88.971| | |基础模型|
|-|text2vec-base-chinese-sentence|36.029|69.853|82.353| | |基础模型|
|-|text2vec-base-chinese-paraphrase|40.441|64.706|78.676| | |基础模型|
|-|text2vec-bge-large-chinese|15.441|36.765|44.853| | |基础模型|
|-|Ernie-model/ernie-1.0-base-zh|2.206|6.618|16.912| | |基础模型|
|-|Ernie-model/ernie-1.0-large-zh-cw|6.618|13.971|18.382| | |基础模型|
|-|Ernie-model/ernie-3.0-medium-zh|0.735|5.147|7.353| | |基础模型|
|-|Ernie-model/ernie-3.0-base-zh|0.0|0.0|0.735| | |基础模型|
|-|Ernie-model/rocketqa-zh-dureader-query-encoder|47.794|78.676|93.382| | |基础模型|
|-|Ernie-model/rocketqa-zh-base-query-encoder|32.353|63.235|72.794| | |基础模型|

## 更新日志

|日期|内容|备注|
|---|---|---|
|20230906|增加open_model向量提取功能<br>增加milvus,faiss,hnswlib向量存储,向量召回功能||