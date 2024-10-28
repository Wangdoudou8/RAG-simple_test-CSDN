 <br>
 
**引言**：之前，我对 RAG 其实也略有了解，不过总是觉得模模糊糊的，这几天好好调研了一下。故此，写下这篇简易版的 RAG技术的 相关内容。





<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

:white_check_mark: 
> 笔者简介：Wang Linyong，西工大，2023级，计算机技术
> 研究方向：文本生成、大语言模型








<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

@[TOC](文章目录)

<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

# 1. 为什么产生RAG技术？

● 因为 RAG 能缓解大型语言模型（Large Language Models，LLMs）现存的一些问题。**那 LLMs 现存哪些问题呢？答：主要是以下 `5` 个：**
1. **计算资源问题：** 因为 LLMs 固有的庞大结构和参数数量使得对其进行修改、微调或重新训练变得异常困难，且相关成本相当显著。
2. **提示依赖问题：** LLMs 的应用往往依赖于构建适当的提示（prompt）来引导模型生成所需的文本。这种方法通过将任务信息嵌入到提示中，从而引导模型按照特定的方向生成文本。然而，这种基于提示的方法可能使模型过于依赖先前见过的模式（即在训练的时候学习过），而无法真正理解问题的本质。
3. **模型幻觉问题：** 幻觉（Hallucination）被定义为 LLMs 生成的内容与提供的源内容无关或不忠实，具体而言，是一种虚假的感知，但在表面上却似乎是真实的。造成幻觉的原因主要可以归结为数据驱动原因、表示和解码的不完善以及参数知识偏见。
4. **时效性问题：** 由于 LLMs 通常是在大量历史数据上进行训练的，这些数据可能包含了过时的信息或者不再准确的事实。当用户询问关于最新事件或者最近发生的变化时，模型可能无法提供最新的信息，因为它的知识库是基于训练数据的，而这些数据可能没有包含最新的发展。
5. **数据安全问题：** 在企业内部每天中会产生大量敏感数据，如客户信息、财务报告、商业战略等。LLMs 应用过程中可能需要访问这些数据，如果将所有的数据都传输给大模型进行问答，可能无意中泄露这些保密信息。





<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

# 2. RAG技术的简要介绍
● **定义：** RAG（Retrieval-Augmented Generation）是一种结合了信息**检索**（Retrieval）、模型能力**增强**（Augmented）和文本**生成**（Generation）的新型 **自然语言处理技术框架**。

● **作用：** RAG 旨在通过**检索**（Retrieval）外部知识库来**增强**（Augmented）大语言模型的能力，从而提高**生成**（Generation）内容的准确性、相关性和丰富性。特别地，它在弥补大型语言模型（LLMs）的局限性方面取得了显著进展，尤其是在解决**幻觉**问题和提升**时效性**方面。

● **使用方式：** RAG 框架的最终输出被设计为一种协同工作模式，即将检索到的知识融合到大型语言模型的生成过程中。

```python
你是一个{task}方面的专家，请结合给定的资料，并回答最终的问题。请如实回答，如果问题在资料中找不到答案，请回答不知道。

问题：{question}

资料：
- {information1}
- {information2}
- ...
```
● 其中，`{task}` 代表任务的领域或主题，`{question}` 是最终要回答的问题（即用户提出的问题文本），而 `{information1}`、`{information2}` 等则是提供给模型的外部知识库。

<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">


# 3. RAG技术和SFT技术的对比

● 模型微调技术（Supervised Fine-Tuning，SFT）是一种在预训练的大语言模型上，利用有标注的特定任务数据对模型进行进一步训练和调整的深度学习策略，旨在提高模型在特定任务上的性能。

● 在更新大型语言模型的知识方面，SFT 与 RAG 这两种方法有着各自的**特点**，**具体如下：**

| 特性 |  程度 | RAG技术 |   程度 |  SFT技术 |
|--|--|--|--|--|
|   **模型知识更新要求**      |  **`低`**   |实时更新检索知识库，适合动态数据，无需频繁重训        |  **`高`**  |  存储静态信息，更新知识需要重新训练      |
|   **数据处理要求**      |  **`低`**    |	Pdf、Excel、Word和Web数据等非结构化知识都支持       |  **`高`** |需构建高质量数据集（结构化知识）       |
|   **模型定制化**    | **`低`**    |专注于信息检索和整合，定制化程度低       |   **`高`** |   可定制模型行为、风格及领域知识，定制化程度高   |
|   **可解释性**     |      **`高`**    |    答案在外部知识库中，可追溯    |     **`低`**    |      答案在模型内部知识库中，不够透明    |
|    **计算资源要求**     |      **`低`**       |   仅需要支持检索的计算资源，显卡要求不高  |   **`高`**    |  	需要训练数据集和微调资源      |
|    **延迟时间**  |      **`长`**       |    数据检索可能增加延迟    |    **`短`**        |     微调后的模型反应更快      |
|    **幻觉减少**     |      **`高`**       |   基于实际数据，模型幻觉减少        |  **`中`**  |     通过特定域训练可减少模型幻觉，但仍然有限  |


● RAG 在利用最新信息、提高可解释性和适应性方面具有明显优势，特别是在需要时效性和准确性的应用场景中。相比之下，微调模型（SFT）可能更适合那些对特定任务有明确优化需求，但对时效性和新信息适应性要求不高的场景。**SFT 与 RAG 的优缺点对比如下：**

|   |   RAG技术 |   SFT技术  |
|--|--|--|
| **优点** |  能利用**最新**信息，提高答案质量，具有更好的可解释性和适应性  |    可针对**特定任务**来优化调整预训练模型  |
| **缺点** | 可能面临检索质量问题和增加额外计算资源的需求 |  更新成本高，对新信息适应性较差  |



<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">


# 4. RAG技术的实现流程

● **RAG技术的实现流程：** 主要包括**问题理解**、**信息检索**和**LLMs调用**三个关键过程。问题理解需要进行一定的数据预处理，优化问题表达；信息检索通过连接外部知识库，获取与问题相关的信息；而 LLMs 调用则用于将这些信息整合到自然语言生成的过程中，以生成最终的回答。**总结如下：**
1. **问题理解：** 准确把握用户的意图。
2. **知识检索：** 从外部知识库中检索相关知识。
3. **答案生成：** 将检索结果与问题结合，并交由 LLMs 进行答案生成。

<br/>

● **RAG技术的挑战**：
1. 在问题理解阶段。如何将用户提问与知识库中的知识**建立**有效的**关联**是一个难点，特别是考虑到用户提问可能模糊，用词不规范，难以直接找到相关的知识。
2. 在知识检索阶段。知识库的**信息来源**可能是**多样**的，包括 PDF、PPT、Excel 等格式。如何有效组织和利用这些非结构化数据具有一定的挑战。
3. 在答案生成阶段。由于 LLMs 的输出可能存在**幻觉问题**，即生成的内容可能与问题不相关，生成准确的回答具有一定难度。

<br/>

● **RAG技术的最新研究：** 在已有的 RAG 研究中，论文综述[「Retrieval-Augmented Generation for Large Language Models: A Survey」（`2024.3.27版` 谷歌引用 `700+`）](https://arxiv.org/pdf/2312.10997) 将 RAG 技术按照复杂度继续划分为了 Naive RAG（初级的RAG），Advanced RAG（高级的RAG）、Modular RAG（模块化的RAG），**三者的概述如下：**
1. **Naive RAG：** RAG技术的最基本形式，也被称为经典 RAG。包括**索引、检索、生成**三个基本步骤。索引阶段将文档库分割成短的知识块（Chunk），并构建向量索引。检索阶段根据问题和 Chunks 的相似度检索相关文档片段。生成阶段以检索到的上下文为条件，生成问题的回答。
2. **Advanced RAG：** 在 Naive RAG 的基础上进行**优化和增强**。包含额外处理步骤，分别在数据索引、检索前和检索后进行。包括更精细的数据清洗、设计文档结构和添加元数据，以提升文本一致性、准确性和检索效率。在检索前使用问题的重写、路由和扩充等方式对齐问题和文档块之间的语义差异。在检索后通过重排序避免 “Lost in the Middle” 现象【`术语解释：` 当相关信息出现在输入上下文的开始或结束时，性能通常是最高的，而当模型必须在长上下文中间访问相关信息时，性能会显著下降。[“Lost in the Middle” 起源的论文链接](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00638/119630)】 ，或通过上下文筛选与压缩缩短窗口长度。
3. **Modular RAG：** 引入更多具体功能模块，例如**查询搜索引擎、融合多个回答**等。技术上融合了检索与微调、强化学习等。流程上对 RAG 模块进行设计和编排，出现多种不同 RAG 模式。提供更大灵活性，系统可以根据应用需求选择合适的功能模块组合。模块化 RAG 的引入使得系统更自由、灵活，适应不同场景和需求。

<br/>

● **RAG技术流程的细节：** 其涉及多个关键模块，每个模块承担着特定的任务，协同工作以实现准确的知识检索和生成自然语言回答。

| 序号 |  技术模块 |  描述  |
|--|--|--|
|  1  |     **意图理解**         |     负责理解用户提出的问题，确定用户的意图和主题。要求处理用户提问的模糊性和不规范性，为后续流程提供清晰的任务目标。      |
|  2    |   **文档解析**    |      负责处理来自不同来源的文件，例如 PDF、PPT、Excel 等。要求将文档内容转化为可处理的结构化形式，为知识检索提供合适的输入。     |
|   3  |    **文档索引**   |   负责将解析后的文档分割成短的知识块（Chunk），并构建向量索引。要求系统能够更快速地找到与用户问题相关的文档片段。  |
|  4   |    **向量嵌入**   |   负责将文档索引中的内容映射为向量表示，以便后续的相似度计算。要求模型能更好地理解文档之间的关系，提高知识检索的准确性。  |
|  5   |   **知识检索**    |    负责根据用户提问和向量嵌入计算的相似度检索或文本检索打分。要求能解决问题和文档之间的语义关联，确保检索的准确性。  |
|   6   |     **重排序**     |    负责在知识检索后对文档库进行重排序。要求能避免 “Lost in the Middle” 现象，确保最相关的文档片段在前面。  |
|   7   |   **答案生成**   |  负责利用 LLMs 生成最终的回答。要求结合检索到的上下文，以生成连贯、准确的文本回答。   |
| 8 | **其他功能**	|  可根据具体应用场景需求引入其他模块化设计的功能模块，如查询搜索引擎、融合多个回答等。 |


<br/>


● **RAG技术的可视化流程：** 下面，我将用一个简单的例子来讲解RAG技术的实现流程。



![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8f9a5dc07aff4a07a4c67cc43f218495.png#pic_center =800x)


● **第一步：问题理解**。用户向 RAG技术框架 输入一个问题 “五四运动的历史意义是什么？”，RAG技术框架将会对其进行（一定的文本预处理，优化文本表达【图中未画出】，然后进行）文本嵌入，得到 <font color=#77ee88>问题文本的嵌入向量</font>。

● **第二步：知识检索**。RAG技术框架将会对外部知识库（包括书、教材、行业规范、操作手册和互联网上的数据等）进行文本提取，得到一系列的字符串，然后将其分割成一块块较短的文本区块（chunk，即知识块），再进行和第一步相同的 文本嵌入，得到 <font color=#0000ff>知识向量库</font>。

● **第三步：答案生成**。RAG技术框架将会对 <font color=#77ee88>第一步得到的问题文本的嵌入向量</font> 和 <font color=#0000ff>第二步得到的知识向量库</font> 进行相似匹配，最终会匹配出 `K` 段和问题相关的知识库原文（即 `K` 个 chunk）。接着，RAG技术框架将会构建一段 提示词Prompt（其内容包括 原问题+知识库原文+一些承接词【例子如前文中的 <u>“2. RAG技术的简要介绍”</u> 的 <u>“使用方式”</u>】），并将其喂入 LLMs，得到对应的文本答案。


<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

# 5. 代码实现

● **实现逻辑：** 
1. 读取 “法律问答数据.csv” 的文本数据（外部知识库），并进行一定的数据预处理。
2. 用户输入原问题，并通过 TF-IDF 算法，从 外部知识库 中筛选出相似度与 原问题 最高的 `3` 条法律内容。
3. 将 原问题 和 筛选出的法律内容 结合起来，一起作为 `prompt` 喂给 LLMs，得到最终的回复。

<br/>

● **<font color=#ff00>注意：** 以下代码中的 `api_key` 和 `服务器 base_url` 需要手动换成你的。

```python
# 自制法律问答 RAG 🤖
import jieba  # 引入jieba库，用于中文分词
import pandas as pd  # 引入pandas库，用于数据处理
from sklearn.feature_extraction.text import TfidfVectorizer  # 引入TfidfVectorizer，用于文本特征提取
import openai  # 引入openai库，用于调用GPT模型


########################  步骤 1(start)  ########################
# 读取法律问答数据
laws_content = pd.read_csv('./data/法律问答数据.csv.zip')
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
    api_key='API key',  # 替换为你的API key
    base_url="代理服务器的 URL",  # 替换为你的代理服务器的 URL
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
```



● **我的输入：** `未成年人无证驾驶怎么处理？`

● **输出结果：**

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/54acc479be904f449f8b45e273b4c3f2.png#pic_center =1000x)

<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">


# 6. 参考文献
<strong>[1]</strong>  [《从零开始动手学RAG：打造个人知识库（保姆级教程）》](https://aistudio.baidu.com/education/group/info/31404)





<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

# 7. 补充说明
● 若有写得不对、欠妥的地方，或有疑问，欢迎评论交流。

● 本项目的 `github` 链接：


<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

:star: :star: 完稿于 2024年10月28日 19:16 教研室工位 💻
