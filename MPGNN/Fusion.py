import csv
import dgl
import torch
from supar import Parser
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.optim as optim
from stanfordcorenlp import StanfordCoreNLP
from transformers import BertTokenizer, BertModel
# from SyntacticGNN.BaseModel.GGNN import GGNNModel
from MPGNN.BaseModel.GGNN import GGNNModel
from MPGNN.BaseModel.GCN import GCNModel
from MPGNN.BaseModel.BiLSTM_MHA import BiLSTM_MultiHeadAttention
from MPGNN.BaseModel.TreeLSTM import TreeLSTM



#Read SST-1 dataset
def read(filepath):
    # 打开CSV文件
    with open(filepath, mode='r') as file:
        # 创建CSV读取器
        reader = csv.reader(file)

        # 假设第一行是标题行，跳过它（如果没有标题行，可以省略这一步）
        next(reader, None)

        # 初始化两个空列表，用于存储输入和输出数据
        train_data = []
        train_label = []

        # 逐行读取CSV文件
        for row in reader:
            # 假设输入位于第一列，输出位于第二列
            input_value = row[0]
            output_value = row[1]

            # 将数据添加到相应的列表中
            train_data.append(input_value)
            train_label.append(output_value)
    return train_data, train_label

filepath = 'C:/Users/Administrator/Desktop/VisionLanguageMABSA/MPGNN/Dataset/SST/train.csv'
train_data, train_label = read(filepath)
# print("Input Data:", train_data[:50])   #train_data[:5]
# print("Output Data:", train_label[:50])   #['3', '2']


#Using Stanford Parsing to get Word and Part-of-speech
nlp = StanfordCoreNLP(r'F:\stanford-corenlp-4.5.5\stanford-corenlp-4.5.5', lang='en')

# 初始化两个空列表，用于存储输入和输出数据
Word_feature = []
POS_feature = []

# 循环遍历每个评论语句
for sentence in train_data:
    # use CoreNLP to part-of-speech
    ann = nlp.pos_tag(sentence)
    # print(nlp.pos_tag(sentence))
    # exaction words and pos
    words = [pair[0] for pair in ann]
    pos_tags = [pair[1] for pair in ann]

    Word_feature.append(words)
    POS_feature.append(pos_tags)

# print(Word_feature)
# print(POS_feature)


#BiLSTM_MultiHeadAttention parameter
input_size_lstm = 768
BiLSTM_hidden_size = 128  # BiLSTM隐藏层大小
BiLSTM_num_layers = 2  # BiLSTM层数
BiLSTM_num_directions = 2  # BiLSTM的方向，双向设置为2
MHA_num_heads = 4  # MultiHeadAttention的头数   256/4 =64

#Tree_LSTM parameter
x_size = 768
h_size = 768
dim = 256

#GGNN model parameter
input_size = 768
hidden_size = 768
num_layers1 = 2
num_steps = 5  # Number of GGNN propagation steps
num_etypes = 1  # Number of edge types (can be adjusted based on your dataset)，多图

#GCN model parameter
input_size2 = 768
hidden_size2 = 768
num_layers2 = 2
num_classes = 5   #标签类别数量

#parameter
output_dim = 1
num_epochs = 50
learning_rate = 0.001

#BiLSTM-MHA initial
bilstm_model = BiLSTM_MultiHeadAttention(input_size_lstm, BiLSTM_hidden_size, BiLSTM_num_layers, BiLSTM_num_directions, MHA_num_heads)

#Tree-LSTM initial
child_sum_tree_lstm = TreeLSTM(x_size=x_size, h_size=h_size, dim=dim, dropout=0.3)

#GGNN initial
ggnn = GGNNModel(input_size, hidden_size, num_layers1, num_steps, num_etypes)

#GCN initial
gcn = GCNModel(input_size2, hidden_size2, num_layers2,num_classes)


# 初始化BERT标记器和模型
# 加载BERT模型和分词器F:\bert-base-cased
model_name = 'F:/bert-base-cased'  # 您可以选择其他预训练模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


# 定义优化器和损失函数
all_parameters = list(bilstm_model.parameters()) + list(child_sum_tree_lstm.parameters()) + list(ggnn.parameters()) + list(gcn.parameters())
optimizer = optim.Adam(all_parameters, lr=learning_rate)
loss_function = nn.CrossEntropyLoss()
number = round(len(train_label) / num_epochs)


total_loss = 0.0  # 用于累积损失值
total_accuracy = 0.0  # 用于累积正确预测的样本数量
for epoch in range(num_epochs):

    bilstm_model.train()
    child_sum_tree_lstm.train()
    ggnn.train()
    gcn.train()

    start_idx = number * epoch
    end_idx = number * (epoch + 1)

    # Word_embedding_feature =  []
    for i, (text, label,pos) in enumerate(zip(Word_feature[start_idx:end_idx], train_label[start_idx:end_idx],POS_feature[start_idx:end_idx])):

        # 将标签转化为张量
        label = int(label)
        label_tensor = torch.tensor([label])

        # 使用BiAffine对句子进行处理得到arcs、rels、probs
        parser = Parser.load('biaffine-dep-en')  # 'biaffine-dep-roberta-en'解析结果更准确
        dataset = parser.predict([text], prob=True, verbose=True)
        # print(dataset.sentences[0])
        # print(f"arcs:  {dataset.arcs[0]}\n"
        #       f"rels:  {dataset.rels[0]}\n"
        #       f"probs: {dataset.probs[0].gather(1, torch.tensor(dataset.arcs[0]).unsqueeze(1)).squeeze(-1)}")


        #获取单词节点特征
        # 标记化句子
        marked_text1 = ["[CLS]"] + text + ["[SEP]"]
        # 将分词转化为词向量
        # tokenized_text = tokenizer.tokenize(marked_text)
        input_ids = torch.tensor(tokenizer.encode(marked_text1, add_special_tokens=True)).unsqueeze(0)  # 添加批次维度
        outputs = model(input_ids)

        # 获取词向量
        word_embeddings = outputs.last_hidden_state

        # 提取单词对应的词向量（去掉特殊标记的部分）
        word_embeddings = word_embeddings[:, 1:-1, :]  # 去掉[CLS]和[SEP]标记
        # 提取单词（去掉特殊标记的部分）
        word_list = [item for item in marked_text1 if item not in ['[CLS]', '[SEP]']]
        # 使用切片操作去除第一个和最后一个元素
        word_embedding_feature = word_embeddings[0][1:-1, :]  # 节点特征


        # 标记化句子
        marked_text2 = ["[CLS]"] + pos + ["[SEP]"]

        # 将分词转化为词向量
        # tokenized_text = tokenizer.tokenize(marked_text)
        input_ids = torch.tensor(tokenizer.encode(marked_text2, add_special_tokens=True)).unsqueeze(0)  # 添加批次维度
        outputs = model(input_ids)

        # 获取词向量
        POS_embeddings = outputs.last_hidden_state

        # 提取单词对应的词向量（去掉特殊标记的部分）
        POS_embeddings = POS_embeddings[:, 1:-1, :]  # 去掉[CLS]和[SEP]标记
        # 提取单词（去掉特殊标记的部分）
        pos_list = [item for item in marked_text2 if item not in ['[CLS]', '[SEP]']]
        # 使用切片操作去除第一个和最后一个元素
        pos_embedding_feature = POS_embeddings[0][1:-1, :]  # 节点特征



        # 获取依存关系特征
        rels = dataset.rels[0]
        # 获取依存特征
        marked_text3 = ["[CLS]"] + rels + ["[SEP]"]
        # 将分词转化为词向量
        input_ids = torch.tensor(tokenizer.encode(marked_text3, add_special_tokens=True)).unsqueeze(0)  # 添加批次维度
        outputs = model(input_ids)

        # 获取词向量
        dep_embeddings = outputs.last_hidden_state

        # 提取单词对应的词向量（去掉特殊标记的部分）
        dep_embeddings = dep_embeddings[:, 1:-1, :]  # 去掉[CLS]和[SEP]标记
        # 提取单词（去掉特殊标记的部分）
        dep_list = [item for item in marked_text3 if item not in ['[CLS]', '[SEP]']]
        # 使用切片操作去除第一个和最后一个元素
        dep_embedding_feature = dep_embeddings[0][1:-1, :]  # 节点特征



        # 使用 torch.cat 进行水平拼接  (n, 768*13)   output1, output2, output3
        syn_feature = torch.cat((word_embedding_feature, pos_embedding_feature, dep_embedding_feature), dim=1)
        max_pooling = nn.MaxPool1d(kernel_size=3)  # 使用3作为池化窗口大小,最大池化
        # 将syn_feature的维度从(n, 768*3)转换为(n, 768)
        syn_feature_pooled = max_pooling(syn_feature)


        # 构建句子的图 g
        arcs = dataset.arcs[0]  # 边的信息
        edges = [i + 1 for i in range(len(arcs))]
        for i in range(len(arcs)):
            if arcs[i] == 0:
                arcs[i] = edges[i]

        # 将节点的序号减一，以便适应DGL graph从0序号开始
        arcs = [arc - 1 for arc in arcs]
        edges = [edge - 1 for edge in edges]
        graph = (arcs, edges)
        syn_graph =torch.tensor(graph)

        # Create a DGL graph
        g = dgl.graph(graph)  # 句子的图结构


        #elimination loop
        # Iterate over arcs and edges to remove common elements
        i = 0
        while i < len(arcs):
            j = 0
            while j < len(edges):
                if arcs[j] == edges[j]:
                    # Remove the common element from both arcs and edges
                    arcs.pop(j)
                    edges.pop(j)
                else:
                    j += 1
            i += 1
        tree_graph = (arcs, edges)
        syn_tree_graph = torch.tensor(tree_graph)
        tree_g = dgl.graph(tree_graph)  # 句子的图结构


        # 训练Word-GNN模型
        optimizer.zero_grad()

        # BiLSTM_MHA model train
        # input size(batch_size, seq_len, input_size)  # batch_size = 32 seq_len = 20 input_size = 768
        reshaped_feature = syn_feature_pooled.unsqueeze(0).expand(1, -1, -1)
        output_bilstm = bilstm_model(reshaped_feature)   #output (1,38,256)
        output_bilstm = output_bilstm.squeeze(0)
        # print(output_bilstm.shape)  #torch.Size([38, 256])


        # Tree_LSTM model train
        # 传入graph ,feature , hidden, cell size (node_num,768)
        output_treelstm = child_sum_tree_lstm(tree_g, syn_feature_pooled, syn_feature_pooled, syn_feature_pooled)
        # print(output_treelstm.shape)  #([38, 256])


        #GGNN model train
        # GGNN模型输入结果为图、节点特征  output.unsqueeze(0)
        output_ggnn = ggnn(g, syn_feature_pooled)
        # print(output_ggnn.shape) torch.Size([38, 768])
        output_ggnn = max_pooling(output_ggnn)   #pooling
        # print(output_ggnn.shape) #torch.Size([38, 256])

        #concat three feature vector
        concat_three_features = torch.cat((output_bilstm, output_treelstm, output_ggnn), dim=1)
        # print(concat_three_features.shape)  #torch.Size([38, 768])

        #GCN模型 torch.Size([n, 5])
        output = gcn(concat_three_features, syn_graph)
        # print(output.shape) #torch.Size([1, 5])


        loss = loss_function(output, label_tensor)  # tensor(1)
        loss.backward()
        optimizer.step()

        # 计算准确率
        _, predicted = torch.max(output, 1)  # 获取最大值的索引为计算的类别
        accuracy = (predicted == label_tensor).sum().item()
        total_accuracy += accuracy  # 累计准确率
        total_loss += loss.item()  # 累积损失值
        # 打印每个 epoch 的损失值
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss}, Accuracy: {accuracy / len(label_tensor)}")

    # 计算并打印每个 epoch 的平均损失值
    average_loss = total_loss / number
    average_accuracy = total_accuracy / number
    with open('mpgnn.txt', 'a') as file:
        # print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}, Average Accuracy: {average_accuracy}")
        output_string = f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}, Average Accuracy: {average_accuracy}\n"
        file.write(output_string)
    total_loss = 0.0
    total_accuracy = 0.0
