import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# 使用 AIHubmix
# 配置 LLM（OpenAI 兼容接口）
Settings.llm = OpenAILike(
    model="glm-4.7-flash-free",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base="https://aihubmix.com/v1",
    is_chat_model=True
)

# Settings.llm = OpenAI(
#     model="deepseek-chat",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_base="https://api.deepseek.com"
# )

# 配置中文嵌入模型，用于构建向量索引
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 读取本地文档
docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

# 基于文档构建向量索引
index = VectorStoreIndex.from_documents(docs)

# 将索引包装成查询引擎
query_engine = index.as_query_engine()

# 打印当前查询引擎使用的提示词模板
print(query_engine.get_prompts())

# 发起查询并输出回答
print(query_engine.query("文中举了哪些例子?"))