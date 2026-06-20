from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Protocol, TypedDict, cast

from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

from build_index import EMBEDDING_MODEL_NAME, get_faiss, split_text

class ChunkEntry(TypedDict):
    id: int
    sha: str
    index: int


class FileEntry(TypedDict):
    sha256: str
    chunks: list[ChunkEntry]


class ChunksData(TypedDict):
    schema_version: int
    index_type: str
    embedding_model: str
    files: dict[str, FileEntry]


class ChunkLookupEntry(TypedDict):
    file: str
    index: int
    sha: str


class EmbeddingsClient(Protocol):
    def embed_query(self, text: str) -> list[float]:
        ...


index: Any | None = None
embeddings: EmbeddingsClient | None = None

INDEX_PATH = Path("index.faiss")
CHUNKS_DATA_PATH = Path("chunks_data.json")
KNOWLEDGE_BASE_DIR = Path("knowledge_base")


@dataclass
class Chunk:
    file: str
    index: int
    text: str
    distance: float


@dataclass
class TextPart:
    text: str
    distance: float

def load_runtime() -> None:
    global index, embeddings
    if index is not None and embeddings is not None:
        return

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )
    embeddings.embed_query("warmup")
    faiss = get_faiss()
    # Work around FAISS/OpenMP crashes seen on recent macOS builds.
    faiss.omp_set_num_threads(1)
    index = faiss.read_index(str(INDEX_PATH))


def load_file_text(file_path: Path) -> str:
    try:
        return file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def load_chunks_data(path: Path = CHUNKS_DATA_PATH) -> ChunksData:
    with path.open("r", encoding="utf-8") as file:
        return cast(ChunksData, json.load(file))


def build_chunk_lookup(chunks_data: ChunksData) -> dict[int, ChunkLookupEntry]:
    lookup: dict[int, ChunkLookupEntry] = {}
    for file_name, file_entry in chunks_data.get("files", {}).items():
        for chunk in file_entry.get("chunks", []):
            chunk_id = int(chunk["id"])
            lookup[chunk_id] = {
                "file": file_name,
                "index": int(chunk["index"]),
                "sha": chunk.get("sha"),
            }
    return lookup


def _find_related(query: str, top_k: int) -> tuple[list[float], list[int]]:
    load_runtime()
    if index is None or embeddings is None:
        raise RuntimeError("Query runtime is not initialized.")
    query_vector = embeddings.embed_query(query)
    distances, ids = index.search(np.array([query_vector], dtype=np.float32), top_k)
    return distances[0].tolist(), [int(item) for item in ids[0].tolist()]


def find_related_chunks(query: str, top_k: int = 10) -> list[Chunk]:
    distances, chunk_ids = _find_related(query, top_k)
    chunks_data = load_chunks_data()
    chunk_lookup = build_chunk_lookup(chunks_data)
    file_text_cache: dict[str, str] = {}
    result: list[Chunk] = []

    for idx, chunk_id in enumerate(chunk_ids):
        if chunk_id < 0:
            continue
        chunk_meta = chunk_lookup.get(chunk_id)
        if chunk_meta is None:
            continue

        file_name = chunk_meta["file"]
        if file_name not in file_text_cache:
            file_text_cache[file_name] = load_file_text(KNOWLEDGE_BASE_DIR / file_name)

        parts = split_text(file_text_cache[file_name])
        local_index = chunk_meta["index"]
        if local_index < 0 or local_index >= len(parts):
            continue
        result.append(
            Chunk(
                file=file_name,
                index=local_index,
                text=parts[local_index],
                distance=float(distances[idx]),
            )
        )

    return result


def find_related_text_parts(query: str, top_k: int = 10) -> list[TextPart]:
    distances, chunk_ids = _find_related(query, top_k)
    chunks_data = load_chunks_data()
    chunk_lookup = build_chunk_lookup(chunks_data)
    file_text_cache: dict[str, str] = {}
    result: list[TextPart] = []

    for idx, chunk_id in enumerate(chunk_ids):
        if chunk_id < 0:
            continue
        chunk_meta = chunk_lookup.get(chunk_id)
        if chunk_meta is None:
            continue

        file_name = chunk_meta["file"]
        if file_name not in file_text_cache:
            file_text_cache[file_name] = load_file_text(KNOWLEDGE_BASE_DIR / file_name)

        parts = split_text(file_text_cache[file_name])
        local_index = chunk_meta["index"]
        if local_index < 0 or local_index >= len(parts):
            continue

        start = max(0, local_index - 1)
        end = min(len(parts), local_index + 2)
        result.append(TextPart(text="".join(parts[start:end]), distance=float(distances[idx])))

    return result


if __name__ == "__main__":
    chunks = find_related_chunks("When did Lyrgal travel to the Fringe Expanse?")
    print(chunks)