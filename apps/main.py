from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
# from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import InMemorySaver

import streamlit as st

output_parser = StrOutputParser()

### data in st session

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


def process_document(path):

    # load the PDF document
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load();

    #split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitted_txt = text_splitter.split_documents(documents=docs)

    # create embeddings for the chunks
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2");
    vectorstore = InMemoryVectorStore.from_documents(
        documents = splitted_txt,
        embedding=embeddings
    )

    # create a agent = tools, llm, system prompt, output parser
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash");

    @tool
    def retrieve_context(query: str):
        """
        Retrieve relevant context to a query from knowledge base.
        """

        context = ""
        docs = vectorstore.similarity_search(query = query, k=2)
        for doc in docs:
            context += doc.page_content + "\n"
        return context


    system_prompt = """You are a helpful assistant that answers questions using retrieved context. 
            My knowledge base consists of the details from the uploaded document. 
            ALWAYS use the `retrieve_context` tool for questions requiring external knowledge."""

    memory_saver = InMemorySaver();


    agents = create_agent(
        tools=[retrieve_context],
        model=llm,
        system_prompt=system_prompt,
        checkpointer = memory_saver
    )

    st.session_state.agent = agents
    st.session_state.document_uploaded = True


# while True:
#     query = input("Enter your question (or type 'exit' to quit): ")
#     if query.lower() == 'exit' :
#         break

#     response = agents.invoke({"messages" : [{"role": "user", "content": query}]},{"configurable" : {"thread_id" : 1}})
#     result = response['messages'][-1].content
#     print("Response:", result);



# upload ui
if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="Upload a PDF document", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        with st.spinner("Uploading and processing the document..."):
            path = "./doc_files/"
            for file in uploaded:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())

            process_document(path)
            st.rerun()

# chat ui

if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)


    query = st.chat_input("Ask anything related to uploaded documents....")
    if query:
        st.session_state.messages.append({"role":"user", "content":query})

        st.chat_message("user").markdown(query)
        response = st.session_state.agent.invoke(
            {"messages":[{"role":"user", "content":query}]},
            {"configurable":{"thread_id":1}}
        )

        answer = output_parser.invoke(response["messages"][-1])
        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({"role":"assistant", "content":answer})