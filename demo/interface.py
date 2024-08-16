import sys, os
sys.dont_write_bytecode = True

import pandas as pd
import streamlit as st
import openai
from streamlit_modal import Modal
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

from llm_agent import ChatBot
from ingest_data import ingest
from retriever import SelfQueryRetriever
import chatbot_verbosity as chatbot_verbosity

import time

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))


st.session_state.api_key = API_KEY
st.session_state.gpt_selection = "gpt-3.5-turbo"
st.session_state.rag_selection = "Generic RAG"

# DATA_PATH = CURRENT_DIR + "/../data/main-data/synthetic-resumes.csv"
DATA_PATH = CURRENT_DIR + "/../data/main-data/resumes1.csv"
FAISS_PATH = CURRENT_DIR + "/../vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

requirements = """
Requirements:
Location:
Candidate's number should start with 07 or +44.
Candidate must be a resident of England.
Qualification:
Candidates must pass the GCSE exam in these countries: UK, IRE, AUS, NZ, CAN, SA.
A tertiary qualification.
Experience:
Classroom experience or formal Teacher training in one of the following countries within the last 2 years: UK, IRE, AUS, NZ, CAN, SA.
Candidate should be a Primary Teacher, Secondary teacher, Teaching assistant, SEN teacher, SEN teaching assistant, LSA, HLTA.
"""

welcome_message = """
  # Introduction 🚀
  This model will help you screen resumes for a job opening.
  
  If your query contains the words 'cv' or 'resume', the following requirements will be applied:
  
  ## Requirements:
  ##### Location:
  Candidate's number should start with 07 or +44.
  Candidate must be a resident of England.
  ##### Qualification:
  Candidates must pass the GCSE exam in these countries: UK, IRE, AUS, NZ, CAN, SA.
  A tertiary qualification.
  ##### Experience:
  Classroom experience or formal Teacher training in one of the following countries within the last 2 years: UK, IRE, AUS, NZ, CAN, SA.
  Candidate should be a Primary Teacher, Secondary teacher, Teaching assistant, SEN teacher, SEN teaching assistant, LSA, HLTA.
  
  Otherwise, the requirements will not be considered.

  ## Example Queries 📋
  Here are some example queries you might use:
1. Find the CV that matches the following criteria NVQ Level 3 in Childcare, and recent experience as a Nursery School Teacher working with SEN children.
  ( 1 will find Applicant no 20)
2. Find the CV of the top English teacher with exceptional qualifications and extensive teaching experience
  ( 2 will find Applicant no 12)
3. Considering the job requirements, which CV among candidates would be the most suitable for the role?
  ( 3 will find Applicant no 13)
4. Can you specify any cv classroom experience or formal teacher training you have completed within the last 2 years in the UK, Ireland, Australia, New Zealand, Canada, or South Africa?
  ( 4 will find Applicant no 10)


"""


info_message = """
  # Information

  ### 1. What if I want to use my own resumes?

  If you want to load in your own resumes file, simply use the uploading button above. 
  Please make sure to have the following column names: `Resume` and `ID`. 

  Keep in mind that the indexing process can take **quite some time** to complete. ⌛

  ### 2. What if I want to set my own parameters?

  You can change the RAG mode and the GPT's model type using the sidebar options above. 

  About the other parameters such as the generator's *temperature* or retriever's *top-K*, I don't want to allow modifying them for the time being to avoid certain problems. 
  FYI, the temperature is currently set at `0.1` and the top-K is set at `5`.  

  ### 3. Is my uploaded data safe? 

  Your data is not being stored anyhow by the program. Everything is recorded in a Streamlit session state and will be removed once you refresh the app. 

  However, it must be mentioned that the **uploaded data will be processed directly by OpenAI's GPT**, which I do not have control over. 
  As such, it is highly recommended to use the default synthetic resumes provided by the program. 

  ### 4. How does the chatbot work? 

  The Chatbot works a bit differently to the original structure proposed in the paper so that it is more usable in practical use cases.

  For example, the system classifies the intent of every single user prompt to know whether it is appropriate to toggle RAG retrieval on/off. 
  The system also records the chat history and chooses to use it in certain cases, allowing users to ask follow-up questions or tasks on the retrieved resumes.
"""

about_message = """
  # About

  This small program is a prototype designed out of pure interest as additional work for the author's Bachelor's thesis project. 
  The aim of the project is to propose and prove the effectiveness of RAG-based models in resume screening, thus inspiring more research into this field.

  The program is very much a work in progress. I really appreciate any contribution or feedback on [GitHub](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline).

  If you are interested, please don't hesitate to give me a star. ⭐
"""


st.set_page_config(page_title="Resume Screening")
st.title("Resume Screening")

if "chat_history" not in st.session_state:
  st.session_state.chat_history = [AIMessage(content=welcome_message)]

if "df" not in st.session_state:
  st.session_state.df = pd.read_csv(DATA_PATH,dtype={'ID': 'Int64'})

if "embedding_model" not in st.session_state:
  st.session_state.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})

if "rag_pipeline" not in st.session_state:
  vectordb = FAISS.load_local(FAISS_PATH, st.session_state.embedding_model, distance_strategy=DistanceStrategy.COSINE, allow_dangerous_deserialization=True)
  st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)

if "resume_list" not in st.session_state:
  st.session_state.resume_list = []



def upload_file():
  modal = Modal(key="Demo Key", title="File Error", max_width=500)
  if st.session_state.uploaded_file != None:
    try:  
      df_load = pd.read_csv(st.session_state.uploaded_file, dtype={'ID': 'Int64'})
      st.session_state.df = df_load
      DEFAULT_DATA_PATH = os.path.join(CURRENT_DIR, "../data/main-data/{}".format(st.session_state.uploaded_file.name))
    except Exception as error:
      with modal.container():
        st.markdown("The uploaded file returns the following error message. Please check your csv file again.")
        st.error(error)
    else:
      if "Resume" not in df_load.columns or "ID" not in df_load.columns:
        with modal.container():
          st.error("Please include the following columns in your data: \"Resume\", \"ID\".")
      else:
        with st.toast('Indexing the uploaded data. This may take a while...'):
          st.session_state.df = df_load
          vectordb = ingest(st.session_state.df, "Resume", st.session_state.embedding_model)
          st.session_state.retriever = SelfQueryRetriever(vectordb, st.session_state.df)
  else:
    st.session_state.uploaded_file_path = DATA_PATH
    st.session_state.df = st.session_state.df.head(20)
    vectordb = FAISS.load_local(FAISS_PATH, st.session_state.embedding_model, distance_strategy=DistanceStrategy.COSINE, allow_dangerous_deserialization=True)
    st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)

def update_user_query(user_query, requirements):
    # Check if 'cv' or 'resume' is in user_query
    if 'cv' in user_query.lower() or 'resume' in user_query.lower():
        user_query = f"{user_query}\n\n{requirements}"
    return user_query

def check_openai_api_key(api_key: str):
  openai.api_key = api_key
  try:
    openai.models.list()
  except openai.AuthenticationError as e:
    return False
  else:
    return True
  
  
def check_model_name(model_name: str, api_key: str):
  openai.api_key = api_key
  model_list = [model.id for model in openai.models.list()]
  return True if model_name in model_list else False


def clear_message():
  st.session_state.resume_list = []
  st.session_state.chat_history = [AIMessage(content=welcome_message)]



user_query = st.chat_input("Type your message here...")

# with st.sidebar:
#   st.markdown("# Control Panel")

#   # st.text_input("OpenAI's API Key", type="password", key="api_key")
#   st.selectbox("RAG Mode", ["Generic RAG", "RAG Fusion"], placeholder="Generic RAG", key="rag_selection")
#   st.text_input("GPT Model", "gpt-3.5-turbo", key="gpt_selection")
#   st.file_uploader("Upload resumes", type=["csv"], key="uploaded_file", on_change=upload_file)
#   st.button("Clear conversation", on_click=clear_message)

#   st.divider()
#   st.markdown(info_message)

#   st.divider()
#   st.markdown(about_message)
#   st.markdown("Made by [Hungreeee](https://github.com/Hungreeee)")


for message in st.session_state.chat_history:
  if isinstance(message, AIMessage):
    with st.chat_message("AI"):
      st.write(message.content)
  elif isinstance(message, HumanMessage):
    with st.chat_message("Human"):
      st.write(message.content)
  else:
    with st.chat_message("AI"):
      message[0].render(*message[1:])


if not st.session_state.api_key:
  st.info("Please add your OpenAI API key to continue. Learn more about [API keys](https://platform.openai.com/api-keys).")
  st.stop()

if not check_openai_api_key(st.session_state.api_key):
  st.error("The API key is incorrect. Please set a valid OpenAI API key to continue. Learn more about [API keys](https://platform.openai.com/api-keys).")
  st.stop()

if not check_model_name(st.session_state.gpt_selection, st.session_state.api_key):
  st.error("The model you specified does not exist. Learn more about [OpenAI models](https://platform.openai.com/docs/models).")
  st.stop()


retriever = st.session_state.rag_pipeline

llm = ChatBot(
  api_key=st.session_state.api_key,
  model=st.session_state.gpt_selection,
)

if user_query is not None and user_query != "":
  with st.chat_message("Human"):
    st.markdown(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))

  with st.chat_message("AI"):
    start = time.time()
    with st.spinner("Generating answers..."):
      user_query = update_user_query(user_query, requirements)
      document_list = retriever.retrieve_docs(user_query, llm, st.session_state.rag_selection)
      query_type = retriever.meta_data["query_type"]
      st.session_state.resume_list = document_list
      stream_message = llm.generate_message_stream(user_query, document_list, [], query_type)
    end = time.time()

    response = st.write_stream(stream_message)
    
    retriever_message = chatbot_verbosity
    retriever_message.render(document_list, retriever.meta_data, end-start)

    st.session_state.chat_history.append(AIMessage(content=response))
    st.session_state.chat_history.append((retriever_message, document_list, retriever.meta_data, end-start))