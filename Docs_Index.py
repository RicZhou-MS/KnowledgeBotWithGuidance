'''
Create Vector Store from all documents in a folder, currently supports .pptx, .docx, .pdf files.

'''

from langchain.document_loaders import (UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader, PyPDFLoader, UnstructuredPDFLoader)
import glob
import langchain.text_splitter as text_splitter
from langchain.text_splitter import (RecursiveCharacterTextSplitter, CharacterTextSplitter)
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()


os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15" #"2022-12-01"
os.environ["OPENAI_API_BASE"] = os.getenv('OPENAI_API_BASE')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

ENGLISH_CHUNK_SIZE = 1400
CHINESE_CHUNK_SIZE = 800

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=ENGLISH_CHUNK_SIZE, chunk_overlap=0)  # chunk_overlap=30

DocPath = "BJDocStore"
files = glob.glob(f"{DocPath}/*.*")

all_docs = []
for p in files:
    if p.lower().endswith(".pptx"):
        loader = UnstructuredPowerPointLoader(p)
        docs = loader.load_and_split(text_splitter)
        print(p)
        print(len(docs))
        all_docs.extend(docs)
    elif p.lower().endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(p)
        docs = loader.load_and_split(text_splitter)
        print(p)
        print(len(docs))
        all_docs.extend(docs)
    elif p.lower().endswith(".pdf"):
        loader = PyPDFLoader(p)
        docs = loader.load_and_split(text_splitter)
        print(p)
        print(len(docs))
        all_docs.extend(docs)


print(len(all_docs))

# vectorstore = FAISS.from_documents(all_docs, OpenAIEmbeddings(chunk_size=1, document_model_name="text-search-curie-doc-001", query_model_name="text-search-curie-query-001")) # text-search-curie-*-001 performance is worse than text-embedding-ada-002
vectorstore = FAISS.from_documents(all_docs, OpenAIEmbeddings(chunk_size=1))
#vectorstore = FAISS.from_documents(all_docs, OpenAIEmbeddings())

FAISS.save_local(vectorstore, DocPath)
