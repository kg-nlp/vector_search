# vector_search
各种向量搜索工具

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

## 功能模块
|-- evaluate.py  评估
|-- data  
|   |-- corpus.tsv   召回库
|   `-- dev.tsv  测试集
|-- model
|   `-- open_model.py  开源模型
|-- tool
    |-- faiss_tool.py
    |-- hnswlib_tool.py
    |-- milvus_tool.py
    `-- vector_extract.py  向量抽取
 


## 更新日志

|日期|内容|备注|
|---|---|---|
|20230906|增加open_model向量提取功能<br>增加milvus,faiss,hnswlib向量存储,向量召回功能||