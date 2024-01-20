import csv
import dgl
import torch
from supar import Parser
import dgl.nn.pytorch as dglnn
import torch.nn as nn
from stanfordcorenlp import StanfordCoreNLP
from transformers import BertTokenizer, BertModel
# from SyntacticGNN.BaseModel.GGNN import GGNNModel



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

filepath = 'F:/Project1/SyntacticGNN/dataset/SST/train.csv'
train_data, train_label = read(filepath)
# print("Input Data:", train_label[:5000])   #train_data[:5]
# print("Output Data:", train_label[:2])   #['3', '2']



#Using Stanford Parsing to get Word and Part-of-speech
nlp = StanfordCoreNLP(r'F:\stanford-corenlp-4.5.5\stanford-corenlp-4.5.5', lang='en')

# 初始化两个空列表，用于存储输入和输出数据
Word_feature = []
POS_feature = []

# 循环遍历每个评论语句
for sentence in train_data:
    # 使用 CoreNLP 进行词性标注
    ann = nlp.pos_tag(sentence)
    # print(nlp.pos_tag(sentence))
    # 提取单词和词性
    words = [pair[0] for pair in ann]
    pos_tags = [pair[1] for pair in ann]

    Word_feature.append(words)
    POS_feature.append(pos_tags)

# print(Word_feature)


#创建GGNN模型
class GGNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_steps, num_etypes):
        super(GGNNModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.ggnn_layers = nn.ModuleList()
        # self.fc_layer = nn.Linear(hidden_size, output_dim)  # 添加全连接层，在联合模型中删除

        for i in range(num_layers):
            if i == 0:
                layer_input_size = input_size
            else:
                layer_input_size = hidden_size
            self.ggnn_layers.append(
                dglnn.GatedGraphConv(layer_input_size, hidden_size, n_steps=num_steps, n_etypes=num_etypes))


    def forward(self, g, node_features):
        for layer in self.ggnn_layers:
            node_features = layer(g, node_features)
        # output = self.fc_layer(node_features)  # 使用全连接层进行调整,在联合模型中删除
        return output

#Word-GNN模型参数
num_nodes = 10
input_size = 768
hidden_size = 768
input_size2 = 768

output_dim = 1
num_layers1 = 3
num_layers2 = 1

num_steps = 5  # Number of GGNN propagation steps
num_etypes = 1  # Number of edge types (can be adjusted based on your dataset)，多图

num_classes = 5   #标签类别数量
num_epochs = 20
learning_rate = 0.001


import torch.optim as optim
from SyntacticGNN.BaseModel.GGNN import GGNNModel
from SyntacticGNN.BaseModel.GCN import GCNModel


#GGNN模型
word_ggnn = GGNNModel(input_size, hidden_size, num_layers1, num_steps, num_etypes)
pos_ggnn = GGNNModel(input_size, hidden_size, num_layers1, num_steps, num_etypes)
dep_ggnn = GGNNModel(input_size, hidden_size, num_layers1, num_steps, num_etypes)

#GCN模型
syn_gcn = GCNModel(input_size2, input_size2, num_layers2,num_classes)

# 定义优化器和损失函数
optimizer = optim.Adam(word_ggnn.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

number = round(len(train_label)/num_epochs)
# print(number)

# 初始化BERT标记器和模型
# 加载BERT模型和分词器F:\bert-base-cased
model_name = 'F:/bert-base-cased'  # 您可以选择其他预训练模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


total_loss = 0.0  # 用于累积损失值
total_accuracy = 0.0  # 用于累积正确预测的样本数量
for epoch in range(num_epochs):

    word_ggnn.train()
    pos_ggnn.train()
    dep_ggnn.train()
    syn_gcn.train()

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

        # 训练Word-GNN模型
        optimizer.zero_grad()

        # Forward pass
        output1 = word_ggnn(g, word_embedding_feature)  # GGNN模型输入结果为图、节点特征  output.unsqueeze(0)
        output2 = pos_ggnn(g, pos_embedding_feature)  # GGNN模型输入结果为图、节点特征  output.unsqueeze(0)
        output3 = dep_ggnn(g, dep_embedding_feature)  # GGNN模型输入结果为图、节点特征  output.unsqueeze(0)

        # 使用 torch.cat 进行水平拼接  (n, 768*13)   output1, output2, output3
        syn_feature = torch.cat((output1, output2, output3), dim=1)

        max_pooling = nn.MaxPool1d(kernel_size=3)  # 使用3作为池化窗口大小,最大池化
        # 将syn_feature的维度从(n, 768*3)转换为(n, 768)
        syn_feature_pooled = max_pooling(syn_feature)

        #GCN模型 torch.Size([n, 5])
        output = syn_gcn(syn_feature_pooled,syn_graph)
        # print(output.shape)

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
    with open('syn_31.txt', 'a') as file:
        # print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}, Average Accuracy: {average_accuracy}")
        output_string = f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}, Average Accuracy: {average_accuracy}\n"
        file.write(output_string)
    total_loss = 0.0
    total_accuracy = 0.0











