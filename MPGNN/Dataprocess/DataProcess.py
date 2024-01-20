import re
from supar import Parser
from stanfordcorenlp import StanfordCoreNLP
import nltk
import os

#Read IMDB dataset
def rm_tags(text):
    re_tag=re.compile(r'<[^>]+>')
    return re_tag.sub('',text)

def read_files(filetype):
    path='C:/Users/Administrator/Desktop/IMDB数据集/aclImdb/'
    file_list=[]
    positive_path=path+filetype+"/pos/"
    for f in os.listdir(positive_path):
        file_list+=[positive_path+f]

    negative_path=path+filetype+"/neg/"
    for f in os.listdir(negative_path):
        file_list+=[negative_path+f]

    print('read',filetype,'files:',len(file_list))
    all_labels=([1]*12500+[0]*12500)
    all_texts=[]
    for fi in file_list:
        with open(fi,encoding='utf8') as file_input:
            all_texts+=[rm_tags("".join(file_input.readlines()))]
    return all_labels,all_texts

train_label, train_data=read_files("train")
test_label, test_data=read_files("test")
# print(train_data)



#Using Stanford Parsing to get Word and Part-of-speech
nlp = StanfordCoreNLP(r'D:\StanfordCoreNLP\stanford-corenlp-4.5.4', lang='en')
sentence = "She saw the video lecture."
print(nlp.pos_tag(sentence))


#train_word文件用于存放单词，train_pos文件用于存放单词对应的词性
with open('E:/PythonProject2/SyntacticGNN/dataset/IMDB/train_word', 'w', encoding='utf-8') as word_file, \
     open('E:/PythonProject2/SyntacticGNN/dataset/IMDB/train_pos', 'w', encoding='utf-8') as pos_file:
    # 循环遍历每个评论语句
    for text in train_data:
        # 使用 CoreNLP 进行词性标注
        ann = nlp.pos_tag(text)

        # 提取单词和词性
        words = [pair[0] for pair in ann]
        pos_tags = [pair[1] for pair in ann]

        # 将单词和词性特征按行写入文件
        word_line = ' '.join(words) + '\n'
        pos_line = ' '.join(pos_tags) + '\n'
        word_file.write(word_line)
        pos_file.write(pos_line)

#test_word文件用于存放单词，test_pos文件用于存放单词对应的词性
with open('E:/PythonProject2/SyntacticGNN/dataset/IMDB/test_word', 'w', encoding='utf-8') as word_file, \
     open('E:/PythonProject2/SyntacticGNN/dataset/IMDB/test_pos', 'w', encoding='utf-8') as pos_file:
    # 循环遍历每个评论语句
    for text in test_data:
        # 使用 CoreNLP 进行词性标注
        ann = nlp.pos_tag(text)

        # 提取单词和词性
        words = [pair[0] for pair in ann]
        pos_tags = [pair[1] for pair in ann]

        # 将单词和词性特征按行写入文件
        word_line = ' '.join(words) + '\n'
        pos_line = ' '.join(pos_tags) + '\n'
        word_file.write(word_line)
        pos_file.write(pos_line)


#train_rels文件用于存放依存关系，train_graph文件用于存放句子转化后的图
with open('E:/PythonProject2/SyntacticGNN/dataset/IMDB/train_rels', 'w', encoding='utf-8') as rels_file, \
     open('E:/PythonProject2/SyntacticGNN/dataset/IMDB/train_graph', 'w', encoding='utf-8') as graph_file:
    # 循环遍历每个评论语句
    for text in train_data:
        sentence = nltk.word_tokenize(text)
        parser = Parser.load('biaffine-dep-en')  # 'biaffine-dep-roberta-en'解析结果更准确
        dataset = parser.predict([sentence], prob=True, verbose=True)
        #依存关系
        rels = dataset.rels[0]

        # 构建句子的图，由弧-->节点
        arcs = dataset.arcs[0]  # 边的信息
        edges = [i + 1 for i in range(len(arcs))]
        for i in range(len(arcs)):
            if arcs[i] == 0:
                arcs[i] = edges[i]

        # 将节点的序号减一，以便适应DGL graph从0序号开始
        arcs = [arc - 1 for arc in arcs]
        edges = [edge - 1 for edge in edges]
        graph = (arcs, edges)

        # 将单词和词性特征按行写入文件
        rels_line = ' '.join(rels) + '\n'
        graph_line = '({}, {})\n'.format(graph[0], graph[1])  # 将图信息转为字符串
        rels_file.write(rels_line)
        graph_file.write(graph_line)
