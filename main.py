import streamlit as st 
import os
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain


# Process uploaded csv file
def process_document(file): 
     with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

     loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                'delimiter': ','})
     document = loader.load()
     
     return document
    
# Query Vector Store similar embeddings
def query_store(query, store):
    response = store.similarity_search(query, k=3)
    contents = [doc.page_content for doc in response]

    return contents

    
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Initialize and return a chat bot
def init_chat_bot():
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    template = """
    You're a cheerful informal customer service guru, assisting business owners with brilliant ideas to elevate their ventures. Follow these guidelines:

    1/ Craft responses similar to past ideas, infused with added intelligence.
    2/ If the business idea isn't in our database, feel free to invent a new one!
    3/ Infuse your creativity with a dash of upbeat humor.
    4/ Ensure your ideas resonate with the business owner's query.
    5/ Keep it concise and easy to grasp.

    Business owner's question:
    {question}

    Past business ideas:
    {ideas}

    Now, dazzle us with your best response!
    """

    prompt = PromptTemplate(input_variables=["question", "ideas"], template=template)
    chat_bot = LLMChain(llm=llm, prompt=prompt)

    return chat_bot



# Queries the chat bot for a response
def query_chat_bot(question, vector_store):
    ideas = query_store(question, vector_store)
    chat_bot = init_chat_bot()
    response = chat_bot.run(question=question, ideas=ideas)

    return response


def main():
    st.sidebar.subheader("Hi, I'm your friendly business idea assistant! 🤖")
    open_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type="csv")
    os.environ["OPENAI_API_KEY"] = open_api_key

    if open_api_key and uploaded_file:
        document = process_document(uploaded_file)
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(document, embeddings)


        # Get user prompt
        if prompt := st.chat_input("How can I help you?"):
                with st.chat_message("user"):
                    st.markdown(prompt)

                st.session_state.messages.append({"role": "user", "content": prompt})

                response = query_chat_bot(prompt, vector_store)

                

                with st.chat_message("assistant"):
                    st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()