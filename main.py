import os
from datetime import datetime
from pypdf import PdfReader 
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# reader = PdfReader("DIABETES.pdf") 
# text = ""
# for i in range(len(reader.pages)): 
#     page = reader.pages[i] 
#     text += page.extract_text()

# # text = text.split("\n")

# with open("Output.txt", "w") as text_file:
#     text_file.write(text)

loader = TextLoader("Output.txt")
# print(text)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    # separators=["\n\n"],
    chunk_size=250,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)

docs = text_splitter.split_documents(documents)

# print(docs)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.from_documents(docs,embeddings)


docs_2 = db.similarity_search(query="what is diabetes?", k=4)

print(docs_2[0])