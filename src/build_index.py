"""Split each repo's guidelines into chunks, embed them locally and store them in Chroma."""
import shutil
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

SRC = Path("data/guidelines")
INDEX_DIR = "data/chroma"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

header_splitter = MarkdownHeaderTextSplitter(
    [("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False)
size_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

docs = []
files = sorted(SRC.glob("*.md"))
for path in files:
    owner, name, kind = path.stem.split("__")
    sections = header_splitter.split_text(path.read_text(encoding="utf-8"))
    for chunk in size_splitter.split_documents(sections):
        chunk.metadata.update({"repo": f"{owner}/{name}", "doc_type": kind})
        docs.append(chunk)
print(f"{len(docs)} chunks from {len(files)} files")

shutil.rmtree(INDEX_DIR, ignore_errors=True)      # rebuild from scratch, never append duplicates
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL,
                                   encode_kwargs={"normalize_embeddings": True})
store = Chroma.from_documents(docs, embeddings, persist_directory=INDEX_DIR,
                              collection_name="guidelines")
print(f"Stored {store._collection.count()} chunks in {INDEX_DIR}")

# Quick retrieval test
hits = store.similarity_search("Do I need to add tests to my pull request?", k=3,
                               filter={"repo": "JabRef/jabref"})
for h in hits:
    print(f"\n--- {h.metadata['doc_type']} | {h.metadata.get('h2', '')}\n{h.page_content[:300]}")