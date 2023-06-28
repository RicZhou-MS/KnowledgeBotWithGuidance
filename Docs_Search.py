from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os

class DocVectorSearch:
    vstore = None

    def __init__(self):
        load_dotenv()
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_VERSION"] = "2023-05-15" #"2022-12-01"
        #os.environ["OPENAI_API_BASE"] = os.getenv('OPENAI_API_BASE')
        #os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
        DocPath = "BJDocStore"
        #DocPath = "HRDocStore"
        self.vstore = FAISS.load_local(DocPath, OpenAIEmbeddings(chunk_size=1))
    
    def getDocs(self, query):
        docs = self.vstore.similarity_search_with_score(query, k=8)
        effectivedocs = []
        for doc in docs:
            # doc[1] is the similarity score, smaller value means more similar
            # doc[0].page_content is the content of the document, if the length is too short, it is not a valid document
            if doc[1] < 0.5 and len(doc[0].page_content) > 20:
                effectivedocs.append(doc[0])
        return effectivedocs[:4] # return top 4 documents
