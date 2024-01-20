import torch
import torch.nn as nn

#Child-sum Tree-LSTM
class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    # 汇聚孩子信息c，还有孩子节点的隐层信息h
    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    # 如何汇聚这里由于叶子节点没有孩子，因此不需要做reduce_func
    def reduce_func(self, nodes):
        #agg childern node vector
        h_tild = torch.sum(nodes.mailbox['h'], 1)
        f = torch.sigmoid(self.U_f(nodes.mailbox['h'])) #forget gate
        c = torch.sum(f * nodes.mailbox['c'], 1)   #candid state
        return {'iou': self.U_iou(h_tild), 'c': c}


    # 汇聚后更新节点表征需要什么操作
    def apply_node_func(self, nodes):
        # 这里的公式(1), (3), (4)中非叶子节点表征是0，因此省略第一项，
        iou = nodes.data['iou'] + self.b_iou
        # 把拼接起来的iou分割开，chunk方法将张量切块
        i, o, u = torch.chunk(iou, 3, 1)   #input gate, output gate, candidate state
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']   #cell state
        h = o * torch.tanh(c)    #output vector
        return {'h': h, 'c': c}


import dgl
from dgl.data.tree import SSTDataset
class TreeLSTM(nn.Module):
    def __init__(self, x_size, h_size, dim, dropout):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, dim)
        self.cell = TreeLSTMCell(x_size, h_size)

    def forward(self, graph, node_feature, h, c):

        g = graph
        feature = node_feature
        g.ndata['iou'] = self.cell.W_iou(self.dropout(feature))
        g.ndata['h'] = h
        g.ndata['c'] = c

        # propagate message and update state
        dgl.prop_nodes_topo(g,
                            message_func=self.cell.message_func,
                            reduce_func=self.cell.reduce_func,
                            apply_node_func=self.cell.apply_node_func)
        # compute logits
        hidden = self.dropout(g.ndata.pop('h'))
        logits = self.linear(hidden)
        return logits

#
# import torch
# import dgl
# import networkx as nx
# import torch.nn as nn
#
# # 创建一个具有层次结构的树
# def create_random_tree(n):
#     G = nx.DiGraph()
#     for i in range(1, n):
#         parent = torch.randint(0, i, (1,))
#         G.add_edge(parent.item(), i)
#     return G
#
#
# # 为了演示，我们创建一个具有10个节点的树
# tree_graph = create_random_tree(38)
# # 将 NetworkX 图转换为 DGL 图
# dgl_tree = dgl.DGLGraph(tree_graph)
#
# # 假设 x_size 和 h_size 是你的模型的输入和隐藏状态的大小
# x_size = 256
# h_size = 768
# dim = 256
#
#
# # 初始化模型
# child_sum_tree_lstm = TreeLSTM(x_size=x_size, h_size=h_size, dim=dim, dropout=0.3)
# print(child_sum_tree_lstm)
#
# # # 假设模型的初始化隐藏状态和细胞状态都是全零的张量
# # # g = dgl_tree      #你需要替换成你的实际图数据
# h_init = torch.randn(dgl_tree.number_of_nodes(), h_size)  #隐藏状态 #
# print(h_init.shape)
# c_init = torch.randn(dgl_tree.number_of_nodes(), h_size)  #细胞状态
#
# # Initialize random feature vectors for each node
# feature = torch.randn(dgl_tree.number_of_nodes(), x_size)
#
# # # 调用模型的 forward 函数
# output_logits = child_sum_tree_lstm(dgl_tree, feature, h_init, c_init)
# print("TreeLSTM model output:",output_logits.shape)


