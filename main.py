import os
import streamlit as st

from dotenv import load_dotenv
from pypdf import PdfReader 
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS

from langchain_classic.chains import LLMChain
from langchain_classic.memory import ConversationBufferMemory

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
# from langchain_community.llms import Ollama 
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint # <-- ADD THIS LINE


from htmltemplates import css, bot_template, user_template


def get_pdf_text(raw_text):
    text = ""                               # initialize empty string to hold all text
    for pdf in raw_text:                    # iterate through each uploaded pdf
        pdf_reader = PdfReader(pdf)         # create a pdf reader object
        for page in pdf_reader.pages:       # iterate through each page
            text += page.extract_text()     # extract text from page and append to text string
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(   # create a text splitter object
        separator="\n",                      # split by new lines
        chunk_size=1000,                     # each chunk will be 1000 characters
        chunk_overlap=200,                   # overlap between chunks
        length_function=len                   # function to calculate length
    )
    chunks = text_splitter.split_text(text)  # split the text into chunks
    return chunks

def get_vector_store(text_chunks):
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)          # create vector store from text chunks
    return vector_store

def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()  # initialize chat model
    llm = HuggingFaceEndpoint( 
        repo_id="mistralai/Mistral-7B-Instruct-v0.1", # <-- Use a text-generation model
        # task="text2text-generation", # <-- This parameter is not valid for HuggingFaceEndpoint
        temperature=0.5,
        max_new_tokens=1024,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)  # create conversation memory
    # Placeholder for actual conversation chain creation
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vectorstore.as_retriever(),
        memory=memory)
    return conversation_chain

def handle_user_question(user_question):
    response = st.session_state.conversation({'question': user_question})  # get response from conversation chain
    st.write(response)  # display bot response
    st.session_state.chat_history = response['chat_history']  # update chat history

    for i, message in enumerate(st.session_state.chat_history): # iterate through chat history 
        if i % 2 == 0: # even index - user message
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True) # display user message
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True) # display bot message


def main():
    load_dotenv()
    st.set_page_config("Chat with PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)  # inject CSS with HTML


    if "conversation" not in st.session_state: # check if conversation state exists
        st.session_state.conversation = None # initialize conversation state so that even if site is refreshed, the conversation is not lost.
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your PDFs :books:")
    user_question = st.text_input("Type your Query here:")

    if user_question:
        handle_user_question(user_question)


    st.write(user_template.replace("{{MSG}}", 'hello robot'), unsafe_allow_html=True)  # display user message template
    st.write(bot_template.replace("{{MSG}}",'hello human '), unsafe_allow_html=True)    # display bot message template

    with st.sidebar:
        st.subheader("Your PDFs")
        st.info("Upload your PDFs here to chat with them.")
        pdf_docs = st.file_uploader("Upload PDFs and click Process", type=["pdf"], accept_multiple_files=True)
        if st.button("Process"):
            st.spinner("Processing...")
            # get pdf text
            raw_text = get_pdf_text(pdf_docs)
            # st.write(raw_text)  # display first 500 characters of extracted text

            # split into chunks
            text_chunks = get_text_chunks(raw_text)  # placeholder for text chunks
            # st.write(text_chunks)  # display number of chunks created

            # create vector store
            vectorstore = get_vector_store(text_chunks)

            # create Conversation Chain
            st.session_state.conversation = get_conversation_chain(vectorstore=vectorstore) # it takes history of the conversation and returns the relevant answers looking at that history


            st.success("Processing complete!")

    

if __name__ == "__main__":
    main()