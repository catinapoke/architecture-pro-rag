import faiss, numpy as np, json
from langchain_huggingface import HuggingFaceEmbeddings
from build_index import split_text

index: faiss.IndexFlatL2 | None = None
embeddings: HuggingFaceEmbeddings | None = None

index = faiss.read_index("index.faiss")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", encode_kwargs={"normalize_embeddings": True})

class Chunk:
    def __init__(self, file: str, index: int, text: str, distance: float):
        self.file = file
        self.index = index
        self.text = text
        self.distance = distance

    def __str__(self):
        return f"Chunk(file={self.file}, index={self.index}, text={self.text}, distance={self.distance})"

class TextPart:
    def __init__(self, text: str, distance: float):
        self.text = text
        self.distance = distance

    def __str__(self):
        return f"TextPart(text={self.text}, distance={self.distance})"

def get_file_by_chunk_index(chunks_data: list[dict], chunk_index: int) -> dict:
    for chunk in chunks_data:
        if chunk['start_index'] <= chunk_index <= chunk['end_index']:
            return chunk
    return None

def get_chunk_text(file_entry: dict, chunk_index: int, file_text: str) -> str:
    local_index = chunk_index - file_entry['start_index']
    return split_text(file_text)[local_index]

def get_text_part_text(file_entry: dict, text_part_index: int, file_text: str) -> str:
    local_index = text_part_index - file_entry['start_index']
    parts = split_text(file_text)
    return ''.join(parts[local_index-1:local_index+1])

def load_file_text(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()

def find_related_chunks(query: str) -> list[Chunk]:
    global index, embeddings
    query_vector = embeddings.embed_query(query)

    # Search the index for the query vector
    distances, indices = index.search(np.array([query_vector], dtype=np.float32), 10)

    chunks = []

    # Load the chunks_data from file
    with open('chunks_data.json', 'r', encoding='utf-8') as file:
        chunks_data = json.load(file)

        for index, item_index in enumerate(indices[0]):
            file = get_file_by_chunk_index(chunks_data, item_index)

            # define chunk text
            file_text = load_file_text('knowledge_base/' + file['file'])
            chunk_text = get_chunk_text(file, item_index, file_text)

            chunks.append(Chunk(file=file['file'], index=item_index, text=chunk_text, distance=distances[0][index]))
    
    return chunks

def find_related_text_parts(query: str) -> list[TextPart]:
    global index, embeddings
    query_vector = embeddings.embed_query(query)

    # Search the index for the query vector
    distances, indices = index.search(np.array([query_vector], dtype=np.float32), 10)

    text_parts = []

     # Load the chunks_data from file
    with open('chunks_data.json', 'r', encoding='utf-8') as file:
        chunks_data = json.load(file)

        for i, item_index in enumerate(indices[0]):
            file = get_file_by_chunk_index(chunks_data, item_index)

            # define chunk text
            file_text = load_file_text('knowledge_base/' + file['file'])
            part_text = get_text_part_text(file, item_index, file_text)

            text_parts.append(TextPart(text=part_text, distance=distances[0][i]))
    

    return text_parts

if __name__ == "__main__":
    chunks = find_related_chunks("When did Lyrgal travel to the Fringe Expanse?")
    print(chunks)