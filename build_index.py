from gliner2 import GLiNER2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
import os, json, time
import faiss
import numpy as np

embeddings: HuggingFaceEmbeddings | None = None
transformer: SentenceTransformer | None = None
index: faiss.IndexFlatL2 | None = None

CHUNK_SIZE = 300
CHUNK_OVERLAP = 60

safety_model = GLiNER2.from_pretrained("fastino/gliguard-LLMGuardrails-300M")
safety_model.to("cpu")  # or "cuda", "mps"

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

def load_file_text(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()

def get_files_list(folder: str) -> list[str]:
    return os.listdir(folder)

def safety_check(message: str) -> bool:
    global safety_model

    JAILBREAK_LABELS = [
        "prompt_injection", "jailbreak_attempt", "policy_evasion",
        "instruction_override", "system_prompt_exfiltration", "data_exfiltration",
         "benign",
    ] # "roleplay_bypass", "hypothetical_bypass", "obfuscated_attack", "multi_step_attack", "social_engineering",

    JAILBREAK_TASK = {
        "labels": JAILBREAK_LABELS,
        "multi_label": True,
        "cls_threshold": 0.4,
    }

    result = safety_model.classify_text(
        message,
        {
            "jailbreak_detection": JAILBREAK_TASK,
        },
        threshold=0.85,
        include_confidence=True,
    )

    dangerous = []

    for item in result["jailbreak_detection"]:
        if item["label"] != "benign" and item["confidence"] >= 0.4:
            dangerous.append(item)

    return len(dangerous) == 0

def pipeline():
    load_models()

    # list of files
    folder = 'knowledge_base'
    files = get_files_list(folder)

    print('got {} files'.format(len(files)))

    chunks_data = []

    chunks_index = 0
    # for each file
    for entry in files:
        # load file text
        text = load_file_text(folder + '/' + entry)
        # split text
        pieces = split_text(text)

        # filter dangerous pieces
        filtered_pieces = []
        for piece in pieces:
            if not safety_check(piece):
                print('dangerous piece')
                continue
            filtered_pieces.append(piece)
        
        pieces = filtered_pieces
        filtered_pieces = None # free memory

        if len(pieces) > 0:
            # encode text
            vectors = encode_text(pieces)

            chunks_start_index = chunks_index
            for piece in pieces:
                chunks_index += 1

            # add data to index
            index.add(vectors)

            # add data to chunks_data
            chunks_data.append({
                'file': entry,
                'start_index': chunks_start_index,
                'end_index': chunks_index - 1,
            })

        print('added {} chunks for file {}'.format(len(pieces), entry))
    
    # save index to file
    faiss.write_index(index, "index.faiss")
    print('saved index to file')

    # save chunks_data to file
    with open('chunks_data.json', 'w', encoding='utf-8') as file:
        json.dump(chunks_data, file)

    print('total chunks length: {}'.format(chunks_index))


def load_models():
    global embeddings, transformer, index

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

    transformer = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    dimension = 768
    index = faiss.IndexFlatL2(dimension)
    print('loaded models!')
    
if __name__ == '__main__':
    time_start = time.time()
    pipeline()
    time_end = time.time()
    print('generated index in {} seconds'.format(time_end - time_start))