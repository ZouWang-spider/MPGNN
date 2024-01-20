from supar import Parser
import nltk
import dgl
import networkx as nx
import torch
import matplotlib.pyplot as plt

#使用BiAffine对句子进行处理得到arcs、rels、probs
text = nltk.word_tokenize('She saw the video lecture.')
print(text)
parser = Parser.load('biaffine-dep-en')   #'biaffine-dep-roberta-en'解析结果更准确
dataset = parser.predict([text], prob=True, verbose=True)
print(dataset.sentences[0])
print(f"arcs:  {dataset.arcs[0]}\n"
      f"rels:  {dataset.rels[0]}\n"
      f"probs: {dataset.probs[0].gather(1,torch.tensor(dataset.arcs[0]).unsqueeze(1)).squeeze(-1)}")

#构建句子的图，由弧-->节点
arcs = dataset.arcs[0]  # 边的信息
edges = [i + 1 for i in range(len(arcs))]
for i in range(len(arcs)):
      if arcs[i] == 0:
            arcs[i] = edges[i]

#将节点的序号减一，以便适应DGL graph从0序号开始
arcs = [arc - 1 for arc in arcs]
edges = [edge - 1 for edge in edges]
graph = (arcs,edges)
graph_line = '({}, {})\n'.format(graph[0], graph[1])  # 将图信息转为字符串
print("graph:", graph)
print(graph_line)

# Create a DGL graph
g = dgl.graph((arcs,edges))
nx.draw(g.to_networkx(),with_labels=True)
plt.show()
