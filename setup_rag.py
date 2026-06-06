import os
import time
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
from docx import Document
from openai import AzureOpenAI
import numpy as np
from langchain_core.embeddings import Embeddings
import pytesseract
from PIL import Image
import io

load_dotenv()

# ── 1. Download documents from Blob ──────────────────────────────────────
def download_blob_documents():
    client = BlobServiceClient(
        os.getenv("BLOB_ACCOUNT_URL"),
        credential=DefaultAzureCredential()
    )
    container = client.get_container_client(os.getenv("BLOB_CONTAINER"))
    os.makedirs("docs/position", exist_ok=True)
    os.makedirs("docs/template", exist_ok=True)
    os.makedirs("docs/historical", exist_ok=True)

    # 下载根目录文件（position PDF + template docx）
    for blob in container.list_blobs(name_starts_with="Contract Reviewer Agent/"):
        filename = blob.name.split("/")[-1]
        if not filename:
            continue
        if "Redacted examples" in blob.name:  # skip subfolders
            continue

        if "Positions" in filename and filename.endswith(".pdf"):
            local_path = f"docs/position/{filename}"
            print(f"Downloaded [position]: {filename}")
        elif filename.endswith(".docx"):
            local_path = f"docs/template/{filename}"
            print(f"Downloaded [template]: {filename}")
        else:
            continue  # other formats skipped

        with open(local_path, "wb") as f:
            f.write(container.get_blob_client(blob.name).download_blob().readall())

    # Download historical contracts
    for blob in container.list_blobs(name_starts_with="Contract Reviewer Agent/Redacted examples/"):
        filename = blob.name.split("/")[-1]
        if not filename:
            continue
        local_path = f"docs/historical/{filename}"
        with open(local_path, "wb") as f:
            f.write(container.get_blob_client(blob.name).download_blob().readall())
        print(f"Downloaded [historical]: {filename}")

# ── 2. Extract text from documents ──────────────────────────────────────────────
def extract_text(filepath):
    if filepath.endswith(".pdf"):
        doc = fitz.open(filepath)
        full_text = []
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if len(text) < 50:  # too few characters, indicating an image page, use OCR
                print(f"  Page {i+1}: using OCR...")
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img)
            full_text.append(text)
        return "\n".join(full_text)
    elif filepath.endswith(".docx"):
        from docx import Document
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return ""

# Create a custom embedding class, replacing AzureOpenAIEmbeddings
class DirectAzureEmbeddings(Embeddings):
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint="https://ai-team-04-hack.openai.azure.com/",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",
        )
        self.model = "text-embedding-3-small"

    def embed_documents(self, texts):
        result = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in result.data]

    def embed_query(self, text):
        result = self.client.embeddings.create(input=[text], model=self.model)
        return result.data[0].embedding
    
# ── 3. Create vector store (Position doc + Historical records) ──────────
def build_vector_store():
    embeddings = DirectAzureEmbeddings()
    
    # Position 文档用更大的 chunk，保留完整的表格行上下文
    position_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    # 其他文档用标准 chunk
    standard_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    all_docs = []
    folders = {
        "docs/position":   ("position", position_splitter),
        "docs/template":   ("template", standard_splitter),
        "docs/historical": ("historical", standard_splitter),
    }

    for folder, (doc_type, splitter) in folders.items():
        for f in os.listdir(folder):
            # 跳过 Word 临时文件
            if f.startswith("~$"):
                continue
            text = extract_text(f"{folder}/{f}")
            if not text.strip():
                continue
            chunks = splitter.create_documents(
                [text],
                metadatas=[{"source": f, "type": doc_type}]
            )
            all_docs.extend(chunks)
            print(f"Indexed [{doc_type}]: {f} ({len(chunks)} chunks)")

    # 一次性建库，避免 add_documents 丢失 metadata 的问题
    import time
    batch_size = 50

    # 按类型分组
    position_docs = [d for d in all_docs if d.metadata.get('type') == 'position']
    template_docs = [d for d in all_docs if d.metadata.get('type') == 'template']
    historical_docs = [d for d in all_docs if d.metadata.get('type') == 'historical']
    print(f"  Position: {len(position_docs)}, Template: {len(template_docs)}, Historical: {len(historical_docs)}")

    def build_store_in_batches(docs, label):
        """分批 embed，用 from_documents 合并，完全不用 add_documents"""
        if not docs:
            return None
        store = None
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            print(f"  [{label}] batch {i//batch_size + 1}/{(len(docs)-1)//batch_size + 1}...")
            if store is None:
                store = FAISS.from_documents(batch, embeddings)
            else:
                # 用 merge_from 代替 add_documents
                new_store = FAISS.from_documents(batch, embeddings)
                store.merge_from(new_store)
            if i + batch_size < len(docs):
                time.sleep(15)
        return store

    print("\nBuilding position store...")
    pos_store = build_store_in_batches(position_docs, "position")
    
    print("Building template store...")
    tmpl_store = build_store_in_batches(template_docs, "template")
    
    print("Building historical store...")
    hist_store = build_store_in_batches(historical_docs, "historical")

    # 合并三个库
    print("\nMerging stores...")
    final_store = pos_store
    if tmpl_store:
        final_store.merge_from(tmpl_store)
    if hist_store:
        final_store.merge_from(hist_store)

    final_store.save_local("vector_store")
    print(f"\nVector store ready!")

if __name__ == "__main__":
    download_blob_documents()
    build_vector_store()
