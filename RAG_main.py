# 自制法律问答 RAG 🤖
import jieba  # 引入jieba库，用于中文分词
import pandas as pd  # 引入pandas库，用于数据处理
from sklearn.feature_extraction.text import TfidfVectorizer  # 引入TfidfVectorizer，用于文本特征提取
import openai  # 引入openai库，用于调用GPT模型


########################  步骤 1(start)  ########################
# 读取法律问答数据
laws_content = pd.read_csv('./data/法律问答数据.csv')
# 过滤掉回复为空的行
laws_content = laws_content[~laws_content['reply'].isnull()]
new_laws_content = []
print("法律内容的条数：", len(laws_content))
for i in range(len(laws_content)):
    # 判断 laws_content['question'][i] 的内容是不是文本
    if isinstance(laws_content['question'].iloc[i], str):
        new_laws_content.append(laws_content['title'].iloc[i] + laws_content['question'].iloc[i] + laws_content['reply'].iloc[i])
    else:
        new_laws_content.append(laws_content['title'].iloc[i] + laws_content['reply'].iloc[i])

# 过滤掉长度小于10的回复(这一步是数据预处理)
new_laws_content = [x for x in new_laws_content if len(x) > 10]
# 取前三条数据作为示例
examples = new_laws_content[:3]
print("前三条法律内容: ", examples)
########################  步骤 1(end)  ########################

########################  步骤 2(start)  ########################
# 读取停用词
stop_words = open("./data/stopwords.txt", "r", encoding='utf-8').readlines()
# 去除每行末尾的换行符
stop_words = [x.strip() for x in stop_words]
print("停用词数量: ", len(stop_words))  # 打印停用词数量

# 初始化TfidfVectorizer
vectorizer = TfidfVectorizer(
    tokenizer=jieba.lcut,  # 使用jieba进行分词
    stop_words=stop_words  # 停用词列表
)
# 获取用户输入的问题
user_question = input("请输入你的问题：")
# 将所有法律内容用来训练模型并提取TF-IDF特征
laws_tfidf = vectorizer.fit_transform(new_laws_content)
# 对用户问题进行TF-IDF特征提取
user_tfidf = vectorizer.transform([user_question])

# 打印法律内容分解出的所有特征名中的最后10个元素
print(vectorizer.get_feature_names_out()[10:])
# 打印TF-IDF矩阵的形状
print("停用词数量:", laws_tfidf.shape)
# 有 31482 个文档，每个文档被表示为一个 41022 维的向量，其中每个元素对应一个独特词汇的TF-IDF值

# 计算用户问题与法律内容的相似度
laws_similarity = user_tfidf.dot(laws_tfidf.T).toarray()[0]
# 获取相似度最高的前三个法律内容的索引
top_laws = [new_laws_content[x] for x in laws_similarity.argsort()[-3:]]

# 打印最相关的三个法律内容
print('最相关的法律内容是: ')
for idx, new in enumerate(top_laws):
    print(f'{idx + 1}. {new}')
########################  步骤 2(end)  ########################

########################  步骤 3(start)  ##########
# 初始化OpenAI客户端
client = openai.OpenAI(
    api_key='sk-ZSSju0J2CpD1CgmNTAPPRH67i47DhgapCbQkaP86VYmge0FO',
    base_url="https://api.openai-proxy.org/v1",
)

# 准备发送给GPT的内容
prompt = f"""请结合相关的资料回答问题：  
我的提问{user_question}  

这是相关的资料：  
{top_laws[0]}  
{top_laws[1]}  
{top_laws[2]}  
"""

# 调用GPT模型进行回答
response = client.chat.completions.create(
    model='gpt-4o-mini',  # 指定GPT模型
    max_tokens=1000,  # 设置最大token数，以限制回复长度
    # stream=True,  # 设置为True以保持对话进行
    messages=[{"role": "user", "content": prompt}]  # 发送的消息
)

# 获取GPT的回复
res = response.choices[0].message.content
# 打印RAG问答机器人的回复
print('\nRAG问答机器人的回复：\n' + res)
########################  步骤 3(end)  ########################


# 初始化OpenAI客户端
client = openai.OpenAI(
    api_key='API key',  # 替换为你的API key
    base_url="代理服务器的 URL",  # 替换为你的代理服务器的 URL
)