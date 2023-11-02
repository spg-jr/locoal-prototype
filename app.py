import streamlit as st
from streamlit_elements import elements, mui, html
from streamlit_extras.metric_cards import style_metric_cards
import streamlit.components.v1 as components

#######################################
# AI Packages (will refactor later)
#######################################
from streamlit_chat import message
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv



import pandas as pd
import json

#######################################
# PAGE SETUP
#######################################
def set_page_config():
    # Set page title and description
    st.set_page_config(page_title="Customer Hub", page_icon=":memo:", layout="wide")

def header(selected):
    if selected == 'Wakefield':
        st.image("/Users/santiagogarcia/Documents/Data Science/Python/Streamlit/Locoal/assets/wakefield.png")
    elif selected == "Black Bison Organics":
        st.image("/Users/santiagogarcia/Documents/Data Science/Python/Streamlit/Locoal/assets/blackBisonOrganics.png")


def sidebar():
    # Add logo to top of sidebar
    st.sidebar.image("/Users/santiagogarcia/Documents/Data Science/Python/Streamlit/Locoal/assets/locoal.png", use_column_width=True)

    # Create drop down box
    selected = sidebar_selectbox()
    
    # Create search box for sidebar menu
    #search_term = st.sidebar.text_input("Search", "")

    return selected


def sidebar_selectbox():
    # Create sidebar selction box
    if 'selection' not in st.session_state:
        st.session_state['selection'] = 0

    selected = st.sidebar.selectbox('Select Client:', ('Wakefield', 'Black Bison Organics'), index=st.session_state['selection'])
    return selected


def footer():
    st.markdown("---")
    st.write("Powered by Tau Builds")
    st.write("View the code on [GitHub](https://github.com/your-github-username/your-repo-name)")


def get_tabs(selected):
    tab1, tab2, tab3, tab4 = st.tabs(["📈Customer Hub", "🗃Dashboard", "🤖Artificial Intelligence", "👽Other"])

    with tab1:
        get_metric_cards(selected)
        get_dataframe()

    with tab2:
        if selected =="Wakefield":
            components.iframe("https://app.powerbi.com/view?r=eyJrIjoiZTAwNGIyYzAtYTJjNi00ZWI2LTg3ZDktOTRhZTBjYzFiZGI3IiwidCI6ImEwNzVjNTUzLTcyNzItNDg4OC1iZDVmLWRjNzNlNWQxODljMyIsImMiOjZ9", height=700)
        elif selected == "Black Bison Organics":
            components.iframe("https://app.powerbi.com/view?r=eyJrIjoiZTAwNGIyYzAtYTJjNi00ZWI2LTg3ZDktOTRhZTBjYzFiZGI3IiwidCI6ImEwNzVjNTUzLTcyNzItNDg4OC1iZDVmLWRjNzNlNWQxODljMyIsImMiOjZ9", height=700)

    with tab3:
        st.header("Ask  Paco 💬")
        store_chat()
        start_session(get_pdf())
    
    with tab4:
        print("Hello World")
        

#######################################
# DATA LOADING
#######################################

@st.cache_data
def load_data():
    df = pd.read_csv("/Users/santiagogarcia/Documents/Data Science/Python/Streamlit/Locoal/assets/TrainingData1.csv")
    return df



#######################################
# VISUALIZATION METHODS
#######################################
def get_metric_cards(selected):
    style_metric_cards(border_left_color="#4aad4a")
    col1, col2, col3, col4 = st.columns(4)

    if selected=="Wakefield":
        col1.metric(label="Gain", value=5000, delta=1000)
        col2.metric(label="Loss", value=5000, delta=-1000)
        col3.metric(label="No Change", value=5000, delta=0)
        col4.metric(label="No Change", value=5000, delta=0)
    elif selected=="Black Bison Organics":
        col1.metric(label="Gain", value=-2000, delta=1000)
        col2.metric(label="Loss", value=3000, delta=-1000)
        col3.metric(label="No Change", value=2000, delta=0)
        col4.metric(label="No Change", value=3000, delta=0)

def get_dataframe():
    with st.expander("Data Preview"):
        st.dataframe(load_data())


################################################
# AI Methods Using Open AI (will refactor later)
################################################
 # function to send/retreive embeddings from language model
def get_embeddings(text):
    load_dotenv()
    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=3000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
      
    # create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    #set language model
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain, knowledge_base


# function to instatiate chat
def store_chat():
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []


# function to upload and extract pdf text
def get_pdf():
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    

# function to begin chat
def start_chat(chain, knowledge_base):
    # request user input
    user_question = st.text_input("Ask a question about your PDF:")
    if user_question:
        docs = knowledge_base.similarity_search(user_question)

        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            print(cb)

        # append user response to chat
        st.session_state.past.append(user_question)
        st.session_state.generated.append(response)


# function to build chat session
def start_session(text):
    if text is not None:
        chain, knowledge_base = get_embeddings(text)
        start_chat(chain, knowledge_base)
    
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')   

#######################################
# RUN APP
#######################################
def main():

    set_page_config()
    selected = sidebar()
    header(selected)
    get_tabs(selected)
    footer()


if __name__ == "__main__":
    main()

