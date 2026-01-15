from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

### chat model
llm_model = ChatOpenAI(model="gpt-5", temperature=0)

### embedding model
embed_model = OpenAIEmbeddings(model="text-embedding-3-large")

# from dotenv import load_dotenv
# from src.models.boeing_chat_model import BoeingChatModel
# from src.models.boeing_embeddings import BoeingEmbeddings
# import os
#
# load_dotenv()
#
# PAT = os.getenv("PAT")
# # print(f"This is {PAT}")
#
# # Chat model
# llm_model = BoeingChatModel(
#     udal_pat=PAT,
#     model="gpt-4o-mini",
# )
#
# # Embedding Model
# embed_model = BoeingEmbeddings(
#     udal_pat=PAT
# )
