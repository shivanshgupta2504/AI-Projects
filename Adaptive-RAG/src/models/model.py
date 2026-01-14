# from dotenv import load_dotenv
# # from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# # from langchain_aws import ChatBedrock
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
#
# load_dotenv()
#
# ### chat model
# # llm_model = ChatOpenAI(temperature=0)
# # llm_model =  ChatBedrock(model_id="anthropic.claude-sonnet-4-20250514-v1:0", region_name="us-west-2", temperature=0)
# llm_model = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0,
# )
#
# ### embedding model
# embed_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


from dotenv import load_dotenv
from src.models.boeing_chat_model import BoeingChatModel
from src.models.boeing_embeddings import BoeingEmbeddings
import os

load_dotenv()

PAT = os.getenv("PAT")
# print(f"This is {PAT}")

# Chat model
llm_model = BoeingChatModel(
    udal_pat=PAT,
    model="gpt-4o-mini",
)

# Embedding Model
embed_model = BoeingEmbeddings(
    udal_pat=PAT
)
