import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_text_splitters import CharacterTextSplitter  
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


# Set API Key
os.environ["OPENAI_API_KEY"] = "your_api_key_here"

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

def split_docs(documents):
    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)

def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def ask_question(vectorstore, query):
    docs = vectorstore.similarity_search(query)
    
    llm = OpenAI()
    
    # Create a prompt template for Q&A
    template = """Use the following context to answer the question.
If you don't know the answer, say you don't know.

Context: {context}
Question: {question}
Answer:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Format the documents into a string
    context_str = "\n".join([doc.page_content for doc in docs])
    
    # Create a chain
    chain = prompt | llm
    
    # Invoke the chain
    response = chain.invoke({
        "context": context_str,
        "question": query
    })
    
    return response

if __name__ == "__main__":
    pdf_path = "sample.pdf"

    print("Loading PDF...")
    documents = load_pdf(pdf_path)

    print("Splitting...")
    chunks = split_docs(documents)

    print("Creating embeddings...")
    vectorstore = create_vector_store(chunks)

    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() == "exit":
            break

        answer = ask_question(vectorstore, query)
        print("\nAnswer:", answer)