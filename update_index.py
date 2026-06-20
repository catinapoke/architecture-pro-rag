from __future__ import annotations

import json
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np

from build_index import (
    CHUNKS_DATA_PATH,
    EMBEDDING_MODEL_NAME,
    INDEX_PATH,
    INDEX_TYPE,
    KNOWLEDGE_BASE_DIR,
    METADATA_SCHEMA_VERSION,
    build_file_metadata,
    create_empty_index,
    get_faiss,
    load_file_text,
    load_models,
    process_file,
)

LOGGER = logging.getLogger("update_index")
LOG_PATH = Path("logs") / "update_index.log"


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

def configure_logging() -> None:
    if LOGGER.handlers:
        return
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def default_chunks_data() -> ChunksData:
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "index_type": INDEX_TYPE,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "files": {},
    }


def load_chunks_data(path: Path = CHUNKS_DATA_PATH) -> ChunksData:
    if not path.exists():
        return default_chunks_data()

    with path.open("r", encoding="utf-8") as file:
        data = cast(ChunksData, json.load(file))
    if data.get("schema_version") != METADATA_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported chunks_data schema: {data.get('schema_version')}. "
            f"Expected {METADATA_SCHEMA_VERSION}."
        )
    if not isinstance(data.get("files"), dict):
        raise ValueError("Invalid chunks_data format: `files` must be an object.")
    return data


def load_or_create_index(path: Path = INDEX_PATH) -> Any:
    if not path.exists():
        return create_empty_index()

    loaded_index = get_faiss().read_index(str(path))
    if not hasattr(loaded_index, "add_with_ids"):
        raise ValueError("index.faiss is not compatible with id-based updates.")
    return loaded_index


def index_files_by_sha(folder: Path = KNOWLEDGE_BASE_DIR) -> dict[str, str]:
    files_sha: dict[str, str] = {}
    if not folder.exists():
        return files_sha
    for file_path in sorted(path for path in folder.iterdir() if path.is_file()):
        files_sha[file_path.name] = _sha_file(file_path)
    return files_sha


def _sha_file(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def find_file_changes(
    stored_files: dict[str, FileEntry],
    current_sha: dict[str, str],
) -> tuple[list[str], list[str], list[str]]:
    stored_names = set(stored_files.keys())
    current_names = set(current_sha.keys())

    new_files = sorted(current_names - stored_names)
    removed_files = sorted(stored_names - current_names)
    changed_files = sorted(
        file_name
        for file_name in (current_names & stored_names)
        if stored_files[file_name].get("sha256") != current_sha[file_name]
    )
    return new_files, changed_files, removed_files


def get_file_chunk_ids(file_entry: FileEntry) -> list[int]:
    return [int(chunk["id"]) for chunk in file_entry.get("chunks", [])]


def merge_file_metadata(
    files_meta: dict[str, FileEntry],
    file_name: str,
    file_metadata: FileEntry | None,
) -> None:
    if file_metadata is None:
        files_meta.pop(file_name, None)
        return
    files_meta[file_name] = file_metadata


def _remove_ids(target_index: Any, ids: list[int]) -> int:
    if not ids:
        return 0
    return int(target_index.remove_ids(np.array(ids, dtype=np.int64)))


def _add_processed_file(target_index: Any, file_path: Path) -> FileEntry:
    processed = process_file(file_path)
    if processed.chunk_ids:
        target_index.add_with_ids(processed.vectors, np.array(processed.chunk_ids, dtype=np.int64))
    return cast(FileEntry, build_file_metadata(processed))


def save_atomically(target_index: Any, chunks_data: ChunksData) -> None:
    index_tmp = INDEX_PATH.with_name(f"{INDEX_PATH.name}.tmp")
    chunks_tmp = CHUNKS_DATA_PATH.with_name(f"{CHUNKS_DATA_PATH.name}.tmp")

    get_faiss().write_index(target_index, str(index_tmp))
    with chunks_tmp.open("w", encoding="utf-8") as file:
        json.dump(chunks_data, file)

    os.replace(index_tmp, INDEX_PATH)
    os.replace(chunks_tmp, CHUNKS_DATA_PATH)


def update_index() -> int:
    configure_logging()
    LOGGER.info("Update started")

    load_models()
    chunks_data = load_chunks_data()
    index_obj = load_or_create_index()
    files_meta: dict[str, FileEntry] = dict(chunks_data.get("files", {}))

    current_sha = index_files_by_sha()
    new_files, changed_files, removed_files = find_file_changes(files_meta, current_sha)
    mode = "no-op"
    if not files_meta and current_sha:
        mode = "initial"
    elif new_files or changed_files or removed_files:
        mode = "incremental"

    LOGGER.info(
        "Mode=%s, new=%s, changed=%s, removed=%s",
        mode,
        new_files,
        changed_files,
        removed_files,
    )

    if mode == "no-op":
        LOGGER.info("No changes detected")
        return 0

    path_by_name = {
        path.name: path
        for path in sorted(path for path in KNOWLEDGE_BASE_DIR.iterdir() if path.is_file())
    } if KNOWLEDGE_BASE_DIR.exists() else {}

    removed_ids_count = 0
    for file_name in removed_files + changed_files:
        old_meta = files_meta.get(file_name)
        if old_meta is None:
            continue
        old_ids = get_file_chunk_ids(old_meta)
        removed_ids_count += _remove_ids(index_obj, old_ids)
        merge_file_metadata(files_meta, file_name, None)

    added_ids_count = 0
    for file_name in new_files + changed_files:
        file_path = path_by_name[file_name]
        new_meta = _add_processed_file(index_obj, file_path)
        merge_file_metadata(files_meta, file_name, new_meta)
        added_ids_count += len(new_meta.get("chunks", []))

    chunks_data["files"] = dict(sorted(files_meta.items()))
    save_atomically(index_obj, chunks_data)
    LOGGER.info(
        "Update finished. removed_chunk_ids=%s, added_chunk_ids=%s, total_vectors=%s",
        removed_ids_count,
        added_ids_count,
        getattr(index_obj, "ntotal", "unknown"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(update_index())
