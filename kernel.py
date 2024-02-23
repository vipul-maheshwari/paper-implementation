import os
import time
import lancedb
from langchain_community.vectorstores import LanceDB

from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, DirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from prettytable import PrettyTable

HF_TOKEN = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# Loading the web URL and breaking down the information into chunks
start_time = time.time()

loader = WebBaseLoader("https://gameofthrones.fandom.com/wiki/Jon_Snow")
documents_loader = DirectoryLoader('data', glob="./*.pdf", loader_cls=PyPDFLoader)

# URL loader
url_docs = loader.load()

# Document loader
data_docs = documents_loader.load()

# Combining all the information into a single variable
docs = url_docs + data_docs

# Specify chunk size and overlap
chunk_size = 256
chunk_overlap = 20
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
chunks = text_splitter.split_documents(docs)

# Specify Embedding Model
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={'device': 'cpu'})

query = "Hello I want to see the length of the embeddings for this document."
embeddings.embed_documents([query])[0]

# Specify Vector Database
vectorstore_start_time = time.time()
database_name = "LanceDB"
db = lancedb.connect("src/lance_database")
table = db.create_table(
    "rag_sample",
    data=[
        {
            "vector": embeddings.embed_query("Hello World"),
            "text": "Hello World",
            "id": "1",
        }
    ],
    mode="overwrite",
)
docsearch = LanceDB.from_documents(chunks, embeddings, connection=table)
vectorstore_end_time = time.time()

# Specify Retrieval Information
search_type = "mmr"
search_kwargs = {"k": 3}
retriever = docsearch.as_retriever(search_kwargs = {"k": 3})

# Specify Model Architecture
llm_repo_id = "huggingfaceh4/zephyr-7b-alpha"
model_kwargs = {"temperature": 0.5, "max_length": 4096, "max_new_tokens": 2048}
model = HuggingFaceHub(repo_id=llm_repo_id, model_kwargs=model_kwargs)

template = """
{query}
"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain_start_time = time.time()
rag_chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
rag_chain_end_time = time.time()

def get_complete_sentence(response):
    last_period_index = response.rfind('.')
    if last_period_index != -1:
        return response[:last_period_index + 1]
    else:
        return response

# Invoke the RAG chain and retrieve the response
rag_invoke_start_time = time.time()
response = rag_chain.invoke("Who killed Jon Snow?")
rag_invoke_end_time = time.time()

# Get the complete sentence
complete_sentence_start_time = time.time()
complete_sentence = get_complete_sentence(response)
complete_sentence_end_time = time.time()

# Create a table
table = PrettyTable()
table.field_names = ["Task", "Time Taken (Seconds)"]

# Add rows to the table
table.add_row(["Vectorstore Creation", round(vectorstore_end_time - vectorstore_start_time, 2)])
table.add_row(["RAG Chain Setup", round(rag_chain_end_time - rag_chain_start_time, 2)])
table.add_row(["RAG Chain Invocation", round(rag_invoke_end_time - rag_invoke_start_time, 2)])
table.add_row(["Complete Sentence Extraction", round(complete_sentence_end_time - complete_sentence_start_time, 2)])

# Additional information in the table
table.add_row(["Embedding Model", embedding_model_name])
table.add_row(["LLM (Language Model) Repo ID", llm_repo_id])
table.add_row(["Vector Database", database_name])
table.add_row(["Temperature", model_kwargs["temperature"]])
table.add_row(["Max Length Tokens", model_kwargs["max_length"]])
table.add_row(["Max New Tokens", model_kwargs["max_new_tokens"]])
table.add_row(["Chunk Size", chunk_size])
table.add_row(["Chunk Overlap", chunk_overlap])
table.add_row(["Number of Documents", len(docs)])


print("\nComplete Sentence:")
print(complete_sentence)

# Print the table
print("\nExecution Timings:")
print(table)
