import time
from langchain_community.llms.ctransformers import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores.faiss import FAISS

model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/db_faiss"

def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type='llama',
        max_new_tokens=1024,
        temperature=0.1
    )
    return llm

def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=['context', 'question'])
    return prompt

def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return llm_chain

def read_vectors_db():
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.load_local(folder_path=vector_db_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
    return db

# Experiments
start_time = time.time()
db = read_vectors_db()
llm = load_llm(model_file)

template = """system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n{context}\nuser\n{question}\nassistant"""

prompt = create_prompt(template)

qa_chain = create_qa_chain(prompt, llm, db)

# Run
# question = "Người chống ung thư nuôi mấy người con?"
# question = "Người dân TPHCM trải bạt kín công viên từ 6h sáng vào ngày giỗ tổ hùng vương phải không?"

# claim = "300 sinh viên biểu tình chống Israel tại Đại học Columbia, Đại học Thành phố New York bị cảnh sát bắt khi biểu tình phản đối chiến sự Gaza"
# claim = "Người bị ung thư nuôi 2 người con không đậu đại học"
# question = claim + " phải không?"
# question = "300 sinh viên biểu tình chống Israel tại Đại học Columbia, Đại học Thành phố New York bị cảnh sát bắt khi biểu tình phản đối chiến sự Gaza phải không?"

# question = "Lên kế hoạch từ sớm là bí quyết đầu tiên cho một chuyến du lịch thành công"
# question = "Chưa có thông tin xác thực về việc các công ty du lịch Đà Lạt đã thiết kế tour theo yêu cầu của từng đoàn khách và liên kết với đơn vị khác để có mức giá tốt nhất, chất lượng đảm bảo"
question = "Roger Federer đang nghỉ dưỡng riêng tư cùng gia đình ở một khu resort tại phường Điện Dương, Quảng Nam"
response = qa_chain.invoke({'query': question})

end_time = time.time()
elapsed_time = end_time - start_time
from pprint import pprint
pprint(response)
pprint(f"Elapsed time: {elapsed_time} seconds")
