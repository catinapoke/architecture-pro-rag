from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Any

from gliner2 import GLiNER2

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import numpy.typing as npt


EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
SAFETY_MODEL_NAME = "fastino/gliguard-LLMGuardrails-300M"
KNOWLEDGE_BASE_DIR = Path("knowledge_base")
INDEX_PATH = Path("index.faiss")
CHUNKS_DATA_PATH = Path("chunks_data.json")
INDEX_DIMENSION = 768
CHUNK_SIZE = 300
CHUNK_OVERLAP = 60
METADATA_SCHEMA_VERSION = 2
INDEX_TYPE = "IndexIDMap2(IndexFlatL2)"

embeddings: Any = None
safety_model: Any = None
index: Any = None


def get_faiss() -> Any:
    import faiss

    return faiss


@dataclass
class ProcessedFile:
    file_name: str
    file_sha256: str
    pieces: list[str]
    vectors: npt.NDArray[np.float32]
    chunk_ids: list[int]
    chunk_shas: list[str]

def split_text(text: str) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return text_splitter.split_text(text)

def encode_text(text: list[str]) -> np.ndarray[float]:
    vectors = embeddings.embed_documents(text)
    return np.array(vectors, dtype=np.float32)


def calculate_sha256(content: str | bytes) -> str:
    payload = content.encode("utf-8") if isinstance(content, str) else content
    return sha256(payload).hexdigest()


def make_chunk_id(file_name: str, chunk_index: int) -> int:
    digest = sha256(f"{file_name}:{chunk_index}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    value &= (1 << 63) - 1
    return value if value != 0 else 1


def load_file_text(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def iter_knowledge_files(folder: Path = KNOWLEDGE_BASE_DIR) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file())


def create_empty_index() -> Any:
    faiss = get_faiss()
    base_index = faiss.IndexFlatL2(INDEX_DIMENSION)
    return faiss.IndexIDMap2(base_index)


def load_models() -> None:
    global embeddings, safety_model, index

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )
    embeddings.embed_query("warmup")

    safety_model = GLiNER2.from_pretrained(SAFETY_MODEL_NAME)
    safety_model.to("cpu")
    
    index = create_empty_index()
    print("loaded models!")


def safety_check(message: str) -> bool:
    jailbreak_labels = [
        "prompt_injection",
        "jailbreak_attempt",
        "policy_evasion",
        "instruction_override",
        "system_prompt_exfiltration",
        "data_exfiltration",
        "benign",
    ]
    jailbreak_task = {
        "labels": jailbreak_labels,
        "multi_label": True,
        "cls_threshold": 0.4,
    }
    result = safety_model.classify_text(
        message,
        {"jailbreak_detection": jailbreak_task},
        threshold=0.85,
        include_confidence=True,
    )
    dangerous = [
        item
        for item in result["jailbreak_detection"]
        if item["label"] != "benign" and item["confidence"] >= 0.4
    ]
    return len(dangerous) == 0


def process_file(file_path: Path) -> ProcessedFile:
    text = load_file_text(file_path)
    pieces = [piece for piece in split_text(text) if safety_check(piece)]
    vectors = encode_text(pieces)
    chunk_ids = [make_chunk_id(file_path.name, idx) for idx, _ in enumerate(pieces)]
    chunk_shas = [calculate_sha256(piece) for piece in pieces]
    return ProcessedFile(
        file_name=file_path.name,
        file_sha256=calculate_sha256(text),
        pieces=pieces,
        vectors=vectors,
        chunk_ids=chunk_ids,
        chunk_shas=chunk_shas,
    )


def build_file_metadata(processed: ProcessedFile) -> dict[str, Any]:
    chunks = [
        {"id": chunk_id, "sha": chunk_sha, "index": idx}
        for idx, (chunk_id, chunk_sha) in enumerate(zip(processed.chunk_ids, processed.chunk_shas))
    ]
    return {
        "sha256": processed.file_sha256,
        "chunks": chunks,
    }


def write_files_to_index(target_index: Any, files: list[Path]) -> dict[str, dict[str, Any]]:
    files_metadata: dict[str, dict[str, Any]] = {}

    for file_path in files:
        processed = process_file(file_path)
        if processed.chunk_ids:
            ids = np.array(processed.chunk_ids, dtype=np.int64)
            target_index.add_with_ids(processed.vectors, ids)
        files_metadata[file_path.name] = build_file_metadata(processed)
        print(f"added {len(processed.pieces)} chunks for file {file_path.name}")

    return files_metadata


def build_chunks_data(files_metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "index_type": INDEX_TYPE,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "files": files_metadata,
    }


def save_index_and_chunks(target_index: Any, chunks_data: dict[str, Any]) -> None:
    faiss = get_faiss()
    faiss.write_index(target_index, str(INDEX_PATH))
    with CHUNKS_DATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(chunks_data, file)


def pipeline() -> None:
    global index
    load_models()
    files = iter_knowledge_files()
    print(f"got {len(files)} files")
    index = create_empty_index()
    files_metadata = write_files_to_index(index, files)
    chunks_data = build_chunks_data(files_metadata)
    save_index_and_chunks(index, chunks_data)
    total_chunks = sum(len(item["chunks"]) for item in files_metadata.values())
    print("saved index to file")
    print(f"total chunks length: {total_chunks}")


if __name__ == "__main__":
    start = time.time()
    pipeline()
    end = time.time()
    print(f"generated index in {end - start} seconds")