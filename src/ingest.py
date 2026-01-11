import os
import glob
import uuid
from tqdm import tqdm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# --- CONFIGURATION ---
DATA_DIR = r"C:\Users\sahil\OneDrive\Desktop\Projects\AI Projects\Scert Chatbot\data\raw_markdown"
COLLECTION_NAME = "scert_bot"
QDRANT_PATH = "qdrant_db"
LOG_FILE = "processed_files.txt"  # Keeps track of progress

def main():
    # 1. INITIALIZE DB & MODEL
    print("🔌 Connecting to Qdrant...")
    client = QdrantClient(path=QDRANT_PATH)
    
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

    print("🧠 Loading Embedding Model...")
    embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')

    # 2. LOAD PROGRESS
    processed_files = set()
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            processed_files = set(f.read().splitlines())
        print(f"🔄 Resuming... Found {len(processed_files)} files already processed.")

    # 3. SETUP SPLITTERS
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4"),
    ])
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", " | ", "。", " "]
    )

    # 4. PROCESSING LOOP
    files = glob.glob(os.path.join(DATA_DIR, "*.md"))
    files_to_process = [f for f in files if os.path.basename(f) not in processed_files]

    if not files_to_process:
        print("✅ All files are already processed!")
        return

    print(f"📂 Processing {len(files_to_process)} new files...")

    # We process file-by-file to ensure we can save progress instantly
    for filepath in tqdm(files_to_process, desc="Ingesting", unit="file"):
        try:
            file_name = os.path.basename(filepath)
            
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # --- Extract Metadata ---
            lines = content.split('\n')
            source_url = "https://www.maa.ac.in/"
            clean_content = content
            if lines and lines[0].startswith("Source:"):
                source_url = lines[0].replace("Source:", "").strip()
                clean_content = "\n".join(lines[1:])

            # --- Splitting ---
            header_splits = markdown_splitter.split_text(clean_content)
            final_chunks = text_splitter.split_documents(header_splits)

            if not final_chunks:
                # Mark empty files as processed so we don't try again
                with open(LOG_FILE, "a", encoding="utf-8") as log:
                    log.write(file_name + "\n")
                continue

            # --- Prepare Batch ---
            texts_to_encode = []
            payloads = []
            ids = []

            for chunk in final_chunks:
                header_context = " > ".join(chunk.metadata.values())
                full_text = f"{header_context}\n{chunk.page_content}"
                
                # 1. Generate Valid UUID (Deterministic based on content)
                # This fixes the "ValueError: badly formed hexadecimal UUID string"
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, full_text + file_name))
                
                ids.append(chunk_id)
                texts_to_encode.append("passage: " + full_text)
                payloads.append({
                    "text": chunk.page_content,
                    "context": header_context,
                    "source_url": source_url,
                    "file_name": file_name
                })

            # --- Encode & Upload (Per File) ---
            # Even if a file has 50 chunks, we encode them all at once
            vectors = embedding_model.encode(texts_to_encode, batch_size=64, show_progress_bar=False)

            points = [
                PointStruct(id=uid, vector=v.tolist(), payload=p)
                for uid, v, p in zip(ids, vectors, payloads)
            ]

            client.upsert(collection_name=COLLECTION_NAME, points=points)

            # --- Save Checkpoint ---
            # We only write to log if upload succeeded
            with open(LOG_FILE, "a", encoding="utf-8") as log:
                log.write(file_name + "\n")

        except Exception as e:
            print(f"\n❌ Error on {os.path.basename(filepath)}: {e}")
            # We do NOT write to log, so this file will be retried next time

    print("\n✅ Ingestion complete.")

if __name__ == "__main__":
    main()