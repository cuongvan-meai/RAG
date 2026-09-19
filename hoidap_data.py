import ollama
import chromadb
from pathlib import Path
from pypdf import PdfReader


# =========================
# 1. ĐỌC FILE TXT / PDF
# =========================

def read_document(file_path):   ## đặt tên file và truy cập vào bằng file_path tức là đường dẫn tới file
    file_path = Path(file_path)  ## chuyển đường dẫn thành Path để dễ dàng kiểm tra

    if file_path.suffix.lower() == ".txt":  ## kiểm tra file có phải là file txt hay không suffix có nghĩa là đuôi file, còn lower() là để chuyển thành chữ in thường
        return file_path.read_text(encoding="utf-8")  ## nếu đúng thì sẽ nhận đọc file, còn utf-8 là để đọc rõ tiếng việt

    elif file_path.suffix.lower() == ".pdf":  # nếu là file PDF thì đi vào phần này
        reader = PdfReader(file_path)  ## nếu đúng thì đọc file bằng PdfReader và lưu vào reader

        text = ""

        for page in reader.pages:  ## duyệt từng trang bằng page trong reafer.pages , .pages là thuộc tính để đọc file
            page_text = page.extract_text() ## đọc toàn bộ data của page

            if page_text:   # nếu có data trong page_text thì đến bước tiếp theo
                text += page_text + "\n"  #thêm các chunk vào text
        return text  # lưu lại kết quả

    else:
        raise ValueError("Chỉ hỗ trợ file TXT hoặc PDF.")  # nếu không được, không có văn bản thì kết thúc vòng lặp


# =========================
# 2. CHUNKING
# =========================

def chunk_text(text, chunk_size=250, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# =========================
# 3. NHẬP ĐƯỜNG DẪN FILE
# =========================

file_path = input("Nhập đường dẫn file TXT hoặc PDF: ")

text = read_document(file_path)

if not text.strip():
    raise ValueError("Không đọc được nội dung tài liệu.")


# =========================
# 4. CHUNK
# =========================

chunks = chunk_text(text)

print("Số chunk:", len(chunks))


# =========================
# 5. EMBEDDING CHUNKS
# =========================

embedding_response = ollama.embed(
    model="nomic-embed-text",
    input=chunks
)

embeddings = embedding_response["embeddings"]


# =========================
# 6. CHROMADB
# =========================

client = chromadb.PersistentClient(
    path="/content/chroma_db"
)

collection = client.get_or_create_collection(
    name="rag_documents"
)

ids = [f"chunk_{i}" for i in range(len(chunks))]

collection.upsert(
    ids=ids,
    documents=chunks,
    embeddings=embeddings
)

print("Đã lưu vào ChromaDB")
print("Số dữ liệu:", collection.count())


# =========================
# 7. HỎI ĐÁP LIÊN TỤC
# =========================

while True:
    question = input("\nNhập câu hỏi (gõ exit để thoát): ")

    if question.lower() == "exit":
        print("Đã thoát.")
        break


    # =========================
    # 8. EMBEDDING CÂU HỎI
    # =========================

    question_response = ollama.embed(
        model="nomic-embed-text",
        input=question
    )

    question_embedding = question_response["embeddings"][0]


    # =========================
    # 9. RETRIEVAL
# =========================

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )


    # =========================
    # 10. CONTEXT
    # =========================

    context = "\n\n".join(
        results["documents"][0]
    )


    # =========================
    # 11. PROMPT
    # =========================

    prompt = f"""
Bạn là trợ lý hỏi đáp tài liệu.

Chỉ trả lời dựa trên Context bên dưới.

Không được tự thêm thông tin không có trong Context.

Nếu Context không có đủ thông tin để trả lời,
hãy trả lời:

\"Không tìm thấy thông tin trong tài liệu.\"

Context:
{context}

Question:
{question}

Answer:
"""


    # =========================
    # 12. QWEN3:4B
    # =========================

    response = ollama.chat(
        model="qwen3:4b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    # =========================
    # 13. ANSWER
    # =========================

    answer = response["message"]["content"]

    print("\nCâu trả lời:")
    print(answer)