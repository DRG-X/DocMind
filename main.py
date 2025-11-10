#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import faiss
import numpy as np

# Text extraction
import pdfplumber
from pdf2image import convert_from_path
from easyocr import Reader

# Embeddings
from sentence_transformers import SentenceTransformer

# LM Studio (OpenAI-compatible client)
from openai import OpenAI

# -------------------------
# Config (edit if needed)
# -------------------------
POPPLER_PATH = r"C:\Users\DEEPANSH  RAJ GOEL\Release-25.07.0-0\poppler-25.07.0\Library\bin"
# If you don't want to hardcode poppler on other machines, set POPPLER_PATH = None

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIR = "vector_index"
FAISS_PATH = os.path.join(VECTOR_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(VECTOR_DIR, "metadata.json")
EMBEDDINGS_JSON = "embeddings.json"
EXTRACTED_TEXTS = "extracted_texts"

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"  # make sure LM Studio server is running
LMSTUDIO_API_KEY = os.environ.get("LM_STUDIO_API_KEY", "lm-studio")  # or set env var

# -------------------------
# Helpers - Stage 1 (text)
# -------------------------
def extract_text_from_pdf(pdf_path):
    """Extract text with pdfplumber. Returns full text (string)."""
    text_content = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    # normalize spaces and strip
                    page_text = " ".join(page_text.split())
                    text_content.append(page_text)
                else:
                    # page probably scanned or empty
                    pass
    except Exception as e:
        print(f"  ⚠️ pdfplumber failed for {pdf_path}: {e}")
    return "\n".join(text_content)

def process_all_pdfs(folder_path, output_folder=EXTRACTED_TEXTS):
    """Extract text from all PDFs in folder_path and save .txt files into output_folder."""
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    for idx, filename in enumerate(files, start=1):
        pdf_path = os.path.join(folder_path, filename)
        print(f"[{idx}/{len(files)}] Extracting text: {filename}")
        text = extract_text_from_pdf(pdf_path)
        out_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
        with open(out_path, "w", encoding="utf-8") as fw:
            fw.write(text)
    print("pdfplumber extraction done.")

# -------------------------
# Helpers - Stage 1b (OCR fallback)
# -------------------------
# initialize EasyOCR reader once
OCR_READER = Reader(['en'], gpu=False)  # set gpu=True if you have CUDA and want to use it

def process_scanned_pdf(pdf_path, output_folder=EXTRACTED_TEXTS, temp_folder="pdf_images"):
    """Convert scanned PDF pages to images and OCR them with EasyOCR; save combined text."""
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(temp_folder, exist_ok=True)

    print(f" Converting PDF -> images for OCR: {os.path.basename(pdf_path)}")
    try:
        pages = convert_from_path(pdf_path, dpi=300, poppler_path=(POPPLER_PATH or None))
    except Exception as e:
        print(f"pdf2image failed: {e}")
        return

    image_paths = []
    for i, page in enumerate(pages):
        image_path = os.path.join(temp_folder, f"page_{i+1}.jpg")
        page.save(image_path, "JPEG")
        image_paths.append(image_path)

    text_lines = []
    for i, img in enumerate(image_paths, start=1):
        print(f"  OCR page {i}/{len(image_paths)} ...")
        try:
            results = OCR_READER.readtext(img, detail=0)
            text_lines.extend(results)
        except Exception as e:
            print(f"EasyOCR failed for {img}: {e}")

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = os.path.join(output_folder, f"{pdf_name}.txt")
    with open(out_path, "w", encoding="utf-8") as fw:
        fw.write("\n".join(text_lines))

    # cleanup temp images
    for p in image_paths:
        try:
            os.remove(p)
        except:
            pass

    print(f"   OCR saved to {out_path}")

# Helpers - Stage 2 (embeddings)
EMBED_MODEL = None

def get_embedding_model():
    global EMBED_MODEL
    if EMBED_MODEL is None:
        print(" Loading embedding model (this may download on first run)...")
        EMBED_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return EMBED_MODEL

def chunk_text(text, chunk_size=1000, overlap=100):
    words = text.split()
    if not words:
        return []
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def generate_embeddings_for_folder(folder, out_json=EMBEDDINGS_JSON):
    """Create embeddings.json from all .txt in folder. Shows progress prints."""
    model = get_embedding_model()
    all_embeddings = []
    txt_files = [f for f in os.listdir(folder) if f.lower().endswith(".txt")]
    if not txt_files:
        print(" No .txt files found in", folder)
        return

    for fi, filename in enumerate(txt_files, start=1):
        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as fr:
            text = fr.read()
        chunks = chunk_text(text)
        print(f"[{fi}/{len(txt_files)}] {filename} → {len(chunks)} chunks")
        for i, chunk in enumerate(chunks, start=1):
            if i % 50 == 0:
                print(f"   • encoding chunk {i}/{len(chunks)} ...")
            emb = model.encode(chunk).astype("float32")
            all_embeddings.append({
                "text": chunk,
                "embedding": emb.tolist(),
                "source": filename
            })
    with open(out_json, "w", encoding="utf-8") as fw:
        json.dump(all_embeddings, fw, ensure_ascii=False)
    print(f" Embeddings saved to {out_json}")

# -------------------------
# Helpers - Stage 3 (FAISS)
# -------------------------
def build_faiss_index(embeddings_json=EMBEDDINGS_JSON, faiss_path=FAISS_PATH, metadata_path=METADATA_PATH):
    """Read embeddings.json, build FAISS index, save index + metadata."""
    os.makedirs(VECTOR_DIR, exist_ok=True)
    print(" Building FAISS index from", embeddings_json)
    with open(embeddings_json, "r", encoding="utf-8") as fr:
        data = json.load(fr)

    embeddings = [item["embedding"] for item in data]
    texts = [item["text"] for item in data]
    sources = [item["source"] for item in data]

    if not embeddings:
        raise RuntimeError("No embeddings found to build FAISS index.")

    embedding_matrix = np.array(embeddings).astype("float32")
    dim = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embedding_matrix)
    faiss.write_index(index, faiss_path)
    print(f" FAISS index saved to {faiss_path} (n={index.ntotal})")

    metadata = [{"text": t, "source": s} for t, s in zip(texts, sources)]
    with open(metadata_path, "w", encoding="utf-8") as fw:
        json.dump(metadata, fw, ensure_ascii=False)
    print(f" Metadata saved to {metadata_path}")

# -------------------------
# Helpers - Stage 4 (Query / Local LLM)
# -------------------------
# Load embeddings model for query (reuse same SentenceTransformer)
def load_faiss_and_metadata(faiss_path=FAISS_PATH, metadata_path=METADATA_PATH):
    if not os.path.exists(faiss_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError("FAISS index or metadata not found. Build index first.")
    index = faiss.read_index(faiss_path)
    with open(metadata_path, "r", encoding="utf-8") as fr:
        metadata = json.load(fr)
    texts = [m["text"] for m in metadata]
    return index, texts

def query_pdf_engine(index, texts, question, top_k=5, model_name="cognitivecomputations-llama-3-8b-instruct-abliterated-v2-smashed"):
    model = get_embedding_model()
    q_emb = model.encode([question]).astype("float32")
    distances, indices = index.search(np.array(q_emb), top_k)
    context_chunks = [texts[i] for i in indices[0] if i < len(texts)]
    context = "\n\n".join(context_chunks)

    prompt = f"""Use ONLY the context below to answer the question accurately and concisely.

CONTEXT:
{context}

QUESTION:
{question}

Answer based only on the context above.
"""
    client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)
    # LM Studio (OpenAI-compatible) chat call
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    # Defensive: some LM Studio versions may return different structure
    try:
        answer = response.choices[0].message.content.strip()
    except Exception:
        # fallback for other shapes
        answer = getattr(response.choices[0].message, "content", str(response))
    return answer

# -------------------------
# Main CLI + flow
# -------------------------
def main():
    print("=== PDF → Embeddings → FAISS → Query (LM Studio) ===\n")
    pdf_folder = input("Enter path to folder containing PDFs: ").strip()
    if not os.path.isdir(pdf_folder):
        print("Folder not found:", pdf_folder)
        sys.exit(1)

    os.makedirs(EXTRACTED_TEXTS, exist_ok=True)

    # Step 1: text extraction (pdfplumber)
    print("\n🔹 Step 1: Extracting text from PDFs (pdfplumber)")
    process_all_pdfs(pdf_folder, EXTRACTED_TEXTS)

    # Step 2: OCR fallback for low-text or missing txt
    print("\n🔹 Step 2: OCR fallback (scanned PDFs)")
    for filename in os.listdir(pdf_folder):
        if not filename.lower().endswith(".pdf"):
            continue
        txt_path = os.path.join(EXTRACTED_TEXTS, f"{os.path.splitext(filename)[0]}.txt")
        if not os.path.exists(txt_path) or os.path.getsize(txt_path) < 500:
            print(f"  {filename} looks scanned/empty → running OCR")
            process_scanned_pdf(os.path.join(pdf_folder, filename), EXTRACTED_TEXTS)

    # Step 3: embeddings generation
    print("\n🔹 Step 3: Generate embeddings")
    generate_embeddings_for_folder(EXTRACTED_TEXTS, EMBEDDINGS_JSON)

    # Step 4: build FAISS index (optionally reuse existing)
    if os.path.exists(FAISS_PATH) and os.path.exists(METADATA_PATH):
        choice = input(f"\nFound existing FAISS index at {FAISS_PATH}. Reuse? (y/n) [y]: ").strip().lower() or "y"
        if choice == "y":
            print("Reusing existing FAISS index.")
        else:
            print("Rebuilding FAISS index from embeddings.json ...")
            build_faiss_index(EMBEDDINGS_JSON, FAISS_PATH, METADATA_PATH)
    else:
        build_faiss_index(EMBEDDINGS_JSON, FAISS_PATH, METADATA_PATH)

    # Load index & metadata for querying
    index, texts = load_faiss_and_metadata(FAISS_PATH, METADATA_PATH)

    # Interactive query loop
    print("\n Pipeline ready. Enter queries (type 'exit' to quit).")
    while True:
        q = input("\nAsk a question: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        try:
            print("⏳ Searching & querying local LLM...")
            ans = query_pdf_engine(index, texts, q)
            print("\n Answer:\n", ans)
        except Exception as e:
            print(" Query failed:", e)

def query_pdf_folder(folder_path, question):
    """
    High-level helper to be called from Streamlit.
    1. Checks if FAISS index and metadata exist, else builds everything.
    2. Loads FAISS + metadata.
    3. Queries the local LLM and returns the answer.
    """
    # Ensure folders exist
    os.makedirs(EXTRACTED_TEXTS, exist_ok=True)
    os.makedirs(VECTOR_DIR, exist_ok=True)

    # STEP 1: Extract text if needed
    if not os.listdir(EXTRACTED_TEXTS):
        print("📝 Extracting text from PDFs...")
        process_all_pdfs(folder_path, EXTRACTED_TEXTS)

    # STEP 2: Generate embeddings if not present
    if not os.path.exists(EMBEDDINGS_JSON):
        print("🧠 Generating embeddings...")
        generate_embeddings_for_folder(EXTRACTED_TEXTS, EMBEDDINGS_JSON)

    # STEP 3: Build FAISS index if not present
    if not (os.path.exists(FAISS_PATH) and os.path.exists(METADATA_PATH)):
        print("📦 Building FAISS index...")
        build_faiss_index(EMBEDDINGS_JSON, FAISS_PATH, METADATA_PATH)

    # STEP 4: Load FAISS + metadata
    index, texts = load_faiss_and_metadata(FAISS_PATH, METADATA_PATH)

    # STEP 5: Run query
    print(f"🔍 Querying PDF engine for: {question}")
    answer = query_pdf_engine(index, texts, question)

    return answer



if __name__ == "__main__":
    main()
