import torch
import dgl.nn.pytorch as dglnn
import torch.nn as nn


#创建GGNN_FC模型
class GGNN_FCModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_steps, num_etypes,num_classes):
        super(GGNN_FCModel, self).__init__()
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

        # 分类层
        self.fc = nn.Linear(hidden_size, num_classes)  # num_classes 是分类的类别数


    def forward(self, g, node_features):
        for layer in self.ggnn_layers:
            node_features = layer(g, node_features)
        # output = self.fc_layer(node_features)  # 使用全连接层进行调整,在联合模型中删除

        # 使用平均池化操作将单词级别的输出汇总为句子级别的输出
        sentence_output = torch.mean(node_features, dim=0, keepdim=True)

        # 通过分类层进行分类
        logits = self.fc(sentence_output)

        return logits



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

        return node_features

