import json
import os
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

json_file_path = "/Users/thinh/Desktop/Docs/Khoa Luan/Data/Merge/merged_file.json"
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

data = load_json(json_file_path)

context = [entry['context'] for entry in data.values()]
# Biến context chứa các đoạn văn bản

# Hàm chunk_text sử dụng RecursiveCharacterTextSplitter để chia các đoạn văn bản
def chunk_text(context_list, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    chunks = []
    for text in context_list:
        chunks.extend(text_splitter.split_text(text))
    
    return chunks

# Chia các đoạn văn bản trong context thành các đoạn nhỏ
chunks = chunk_text(context)

# Tạo embeddings cho các đoạn nhỏ
def create_embeddings(chunks, model_file):
    embedding_model = GPT4AllEmbeddings(model_file=model_file)
    embeddings = embedding_model.embed_documents(chunks)
    return embeddings, embedding_model

# Đường dẫn đến mô hình và cơ sở dữ liệu vector
model_file = "models/all-MiniLM-L6-v2-f16.gguf"
vector_db_path = "vectorstores/db_faiss"

# Tạo embeddings cho các đoạn nhỏ
embeddings, embedding_model = create_embeddings(chunks, model_file)

# Lưu các embeddings vào cơ sở dữ liệu FAISS
def store_in_faiss(chunks, embedding_model, vector_db_path):
    if not os.path.exists(vector_db_path):
        os.makedirs(vector_db_path)
    db = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    db.save_local(vector_db_path)

# Lưu vào FAISS
store_in_faiss(chunks, embedding_model, vector_db_path)

print("Vector database created successfully.")
