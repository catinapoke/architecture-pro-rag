import faiss, numpy as np, json
from langchain_huggingface import HuggingFaceEmbeddings
from build_index import split_text

def get_file_by_chunk_index(chunks_data: list[dict], chunk_index: int) -> dict:
    for chunk in chunks_data:
        if chunk['start_index'] <= chunk_index <= chunk['end_index']:
            return chunk
    return None

def get_chunk_text(file_entry: dict, chunk_index: int, file_text: str) -> str:
    local_index = chunk_index - file_entry['start_index']
    return split_text(file_text)[local_index]

def load_file_text(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()

# Load the index from a local file
index = faiss.read_index("index.faiss")

embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

query_vector = embeddings.embed_query("When did Lyrgal travel to the Fringe Expanse?")

# Search the index for the query vector
distances, indices = index.search(np.array([query_vector], dtype=np.float32), 10)

# Load the chunks_data from file
with open('chunks_data.json', 'r', encoding='utf-8') as file:
    chunks_data = json.load(file)

    for index, item_index in enumerate(indices[0]):
        file = get_file_by_chunk_index(chunks_data, item_index)

        # define chunk text
        file_text = load_file_text('knowledge_base/' + file['file'])
        chunk_text = get_chunk_text(file, item_index, file_text)

        print('file: {} with chunk start index {} and end index {} got distance {} to chunk index {} with text: {}'.format(file['file'], file['start_index'], file['end_index'], distances[0][index], item_index, chunk_text))
