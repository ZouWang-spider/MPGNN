import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_directions):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True)
        self.num_directions = num_directions

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return lstm_out

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_size // num_heads

        self.fc_query = nn.Linear(input_size, input_size)
        self.fc_key = nn.Linear(input_size, input_size)
        self.fc_value = nn.Linear(input_size, input_size)
        self.fc_out = nn.Linear(input_size, input_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Linear transformations for query, key, and value
        query = self.fc_query(x)
        key = self.fc_key(x)
        value = self.fc_value(x)

        # Split the transformed features into multiple heads
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Calculate attention scores
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention to the value
        output = torch.matmul(attention_weights, value).permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, seq_len, -1)

        # Linear transformation for the output
        output = self.fc_out(output)
        return output

class BiLSTM_MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_directions, num_heads):
        super(BiLSTM_MultiHeadAttention, self).__init__()
        self.bilstm = BiLSTM(input_size, hidden_size, num_layers, num_directions)
        self.multihead_attention = MultiHeadAttention(hidden_size * num_directions, num_heads)

    def forward(self, x):
        lstm_out = self.bilstm(x)
        attention_out = self.multihead_attention(lstm_out)
        return attention_out



# # 使用示例
# input_size = 768  # 输入特征的维度
# hidden_size = 128  # BiLSTM隐藏层大小
# num_layers = 2  # BiLSTM层数
# num_directions = 2  # BiLSTM的方向，双向设置为2
# num_heads = 4  # MultiHeadAttention的头数   256/4 =64
#
# model = BiLSTM_MultiHeadAttention(input_size, hidden_size, num_layers, num_directions, num_heads)


# import torch
#
# # 创建随机输入张量
# batch_size = 32
# seq_len = 20
# input_size = 768
#
# random_input = torch.randn(batch_size, seq_len, input_size)
#
# # 创建 BiLSTM_MultiHeadAttention 模型
# hidden_size = 128
# num_layers = 2
# num_directions = 2
# num_heads = 4
#
# model = BiLSTM_MultiHeadAttention(input_size, hidden_size, num_layers, num_directions, num_heads)
#
# # 将随机输入传递给模型
# output = model(random_input)
#
# # 打印输出张量的形状
# print("Output Shape:", output.shape)  ([32, 20, 256])


