import os
import glob
from tqdm import tqdm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# --- CONFIGURATION ---
DATA_DIR = r"C:\Users\sahil\OneDrive\Desktop\Projects\AI Projects\Scert Chatbot\data\raw_markdown"
COLLECTION_NAME = "scert_bot"
QDRANT_PATH = "qdrant_db"
BATCH_SIZE = 64  # Optimized for 16GB RAM (Safe range: 32-128)

def main():
    # 1. INITIALIZE DATABASE & MODEL
    print("🔌 Connecting to Qdrant...")
    client = QdrantClient(path=QDRANT_PATH)
    
    # Create collection if it doesn't exist
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

    print("🧠 Loading Embedding Model (intfloat/multilingual-e5-large)...")
    # This might take a moment to load into RAM
    embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')

    # 2. SETUP SPLITTERS
    # Split by Markdown headers first to preserve structure
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ])
    
    # Then split purely by characters for large sections
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", " | ", "。", " "]
    )

    # 3. LOAD & CHUNK FILES (Stage 1)
    files = glob.glob(os.path.join(DATA_DIR, "*.md"))
    if not files:
        print(f"❌ No markdown files found in '{DATA_DIR}'. Please check the path.")
        return

    all_chunks_data = [] # Buffer to hold processed text before encoding
    
    print(f"\n📂 Step 1: Processing {len(files)} files...")
    
    for file_index, filepath in enumerate(tqdm(files, desc="Reading Files", unit="file")):
        try:
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

            # --- Buffer Data ---
            for chunk_index, chunk in enumerate(final_chunks):
                # Context string (e.g. "History > Establishment")
                header_context = " > ".join(chunk.metadata.values())
                full_text = f"{header_context}\n{chunk.page_content}"
                
                all_chunks_data.append({
                    "id": f"{file_index}_{chunk_index}",
                    "full_text": full_text,  # Text to embed
                    "payload": {             # Data to store
                        "text": chunk.page_content,
                        "context": header_context,
                        "source_url": source_url,
                        "file_name": os.path.basename(filepath)
                    }
                })

        except Exception as e:
            print(f"⚠️ Error reading {os.path.basename(filepath)}: {e}")

    # 4. BATCH ENCODING (Stage 2)
    total_chunks = len(all_chunks_data)
    print(f"\n⚡ Step 2: Encoding {total_chunks} chunks (Batch Size: {BATCH_SIZE})...")
    
    # Prepare texts with the "passage:" prefix required by E5 models
    texts_to_encode = ["passage: " + item["full_text"] for item in all_chunks_data]
    
    # Encode all at once (fastest method on CPU/GPU)
    # This automatically shows a progress bar
    vectors = embedding_model.encode(
        texts_to_encode, 
        batch_size=BATCH_SIZE, 
        show_progress_bar=True
    )

    # 5. UPLOAD TO QDRANT (Stage 3)
    print(f"\n💾 Step 3: Uploading to Database...")
    
    points_batch = []
    
    # We upload in batches of 100 to avoid locking the DB
    for i, (data, vector) in enumerate(zip(all_chunks_data, vectors)):
        points_batch.append(
            PointStruct(
                id=data["id"],
                vector=vector.tolist(),
                payload=data["payload"]
            )
        )
        
        # Upload when batch is full or at the very end
        if len(points_batch) >= 100 or i == total_chunks - 1:
            client.upsert(collection_name=COLLECTION_NAME, points=points_batch)
            points_batch = [] # Reset batch

    print(f"✅ Success! Indexed {total_chunks} chunks into '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main()