Lưu ý trước khi chạy:
- chạy trên google collab
- khi chạy chú ý cài lại các thư viện :
   !pip install ollama
   !pip install chromadb 
   !pip install pypdf
- chỉ là data ví dụ để chạy chưa là data lớn
- khi chạy với data mới thì làm lại các bước này:
+ 1: Ollama
!curl -fsSL https://ollama.com/install.sh | sh
import subprocess, time
subprocess.Popen(["ollama", "serve"])
time.sleep(5)
!ollama pull nomic-embed-text
!ollama pull qwen3:4b

+ 2: Thư viện
!pip install ollama chromadb pypdf

+ 3: Drive
from google.colab import drive
drive.mount('/content/drive')

+ 4: dán code chính vào, nhớ sửa dòng đường dẫn database thành /content/drive/MyDrive/tên file.
