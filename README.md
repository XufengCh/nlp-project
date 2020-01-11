# 基于Seq2Seq Attention的机器自动回复

复旦大学计算机学院2019-2020学年秋季学期自然语言处理 课程项目

16307130080 陈旭锋

## 环境依赖

> tensorflow 2.0.0
>
> jieba
>
> gensim
>
> numpy
>
> sklearn

## 文件组织

> data_util.py：数据集处理
>
> seq3seq_model.py：定义了Encoder，Decoder模型
>
> seq2seq.ini：配置文件
>
> train_word2vec.py：训练word2vec模型
>
> execute.py：用于模型的训练、测试
>
> ./dataset/：用于存放数据集
>
> ./report/：存放项目报告
>
> ./word2vec/：存放训练好的word2vec模型

## 运行说明

### 训练模型

在配置文件seq2seq.ini中设置

> mode = train

并将word2vec模型，数据集放入对应的路径，再运行

> python execute.py

训练过程中支持使用tensorflow查看训练情况，默认日志文件存放地址为./logs

### 测试模型

训练完成后，在配置文件seq2seq.ini中设置

> mode = test

再运行

> python execute.py

即可测试模型实际效果。输入exit即可退出测试。

## 项目实现细节

请参考./report文件夹下的项目报告