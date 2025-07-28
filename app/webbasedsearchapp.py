import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")


if 'current_link' not in st.session_state:
    st.session_state.current_link = None

if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None

if 'llm' not in st.session_state:
    st.session_state.llm = None


# i need a user for the link 
link = st.text_input("Enter the url to ask about: ")
if link and (st.session_state.current_link != link):
    # start data injestion process
        from langchain_community.document_loaders import WebBaseLoader
        loader = WebBaseLoader(link, requests_kwargs={"headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }})


        docs = loader.load()

        # data transforming

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)

        documents = text_splitter.split_documents(docs)

        # create embeddings
        from langchain_ollama import OllamaEmbeddings
        from langchain_ollama import OllamaLLM

        st.session_state.llm = OllamaLLM(model="llama3.2:1b")
        embeddings = OllamaEmbeddings(model="llama3.2:1b")

        #vector store db
        from langchain_community.vectorstores import FAISS
        st.session_state.vector_db = FAISS.from_documents(documents,embeddings)

       
        st.session_state.current_link = link
        st.success("New Database created sucessfully")
        
        
# question answering section with fresh retrieval for each question       
if st.session_state.vector_db is not None:
            user_query = st.text_input("Ask a Question about this page")
            
            
            
            if user_query:
                retriever = st.session_state.vector_db.as_retriever(
                    search_type="mmr",
                search_kwargs={'k': 6, 'lambda_mult': 0.25}
                )
                with st.expander("Show retrieved context"):
                    relevant_docs = retriever.invoke(user_query)
                    
                    for doc in relevant_docs:
                        st.write(doc.page_content[:])
                
                
                # creating fresh retreival chain each time
                
                from langchain_core.prompts import ChatPromptTemplate
                from langchain.chains.combine_documents import create_stuff_documents_chain
                from langchain.chains import create_retrieval_chain
                
                
                prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the given context below.
    Question: {input}
    
    <context>
    {context}
    </context>
    
    """
                )
                
                #new retrieval every time
                document_chain = create_stuff_documents_chain(st.session_state.llm, prompt)
               
                retrieval_chain = create_retrieval_chain(retriever,document_chain)
                
                #debugging for what context it has used
                
               
                        
                # we get the answer
                response = retrieval_chain.invoke({"input": user_query})  
                st.write('## Answer :')
                st.write(response["answer"])        

