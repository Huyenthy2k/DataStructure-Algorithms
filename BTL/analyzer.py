import os
import json
import pickle
from collections import Counter, defaultdict
from tqdm import tqdm
import torch
from transformers import pipeline, AutoTokenizer

class IndexManager:
    def __init__(self):
        self.inverted_index = defaultdict(set)
        self.entity_frequency = Counter()
        self.co_occurrence = defaultdict(Counter)
    
    def add_document(self, file_path, entities):
        # entities is a list of (entity_text, entity_type)
        unique_entities = set([e[0] for e in entities])
        
        for entity_text, _ in entities:
             self.inverted_index[entity_text].add(file_path)
             self.entity_frequency[entity_text] += 1
             
        unique_list = list(unique_entities)
        for i in range(len(unique_list)):
            e1 = unique_list[i]
            for j in range(i + 1, len(unique_list)):
                e2 = unique_list[j]
                self.co_occurrence[e1][e2] += 1
                self.co_occurrence[e2][e1] += 1

    def save(self, path='index.pkl'):
        # Convert to standard dicts
        with open(path, 'wb') as f:
            pickle.dump({
                'inverted_index': dict(self.inverted_index),
                'entity_frequency': self.entity_frequency,
                'co_occurrence': dict(self.co_occurrence)
            }, f)
        print(f"Index saved to {path} with {len(self.inverted_index)} entities.")

    def load(self, path='index.pkl'):
        pass

class DataLoader:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def get_files(self):
        file_paths = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".txt"):
                    file_paths.append(os.path.join(root, file))
        return file_paths

    def parse_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
                data = json.loads(content)
                text = f"{data.get('Subject', '')} . {data.get('Summary', '')} . {data.get('Content', '')}"
                return text
        except Exception:
            return ""

def main():
    root_dir = "/Users/nguyensiry/Documents/Code_practice/Data_structure_algorithms/BTL/Article_Crawl"
    
    print("Initializing Deep Learning Model...")
    device = -1
    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        device = "mps" 
        
    print(f"Using device: {device}")
    
    model_name = "NlpHUST/ner-vietnamese-electra-base"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        ner_pipe = pipeline("ner", model=model_name, 
                            aggregation_strategy="simple", 
                            device=device)
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return

    loader = DataLoader(root_dir)
    files = loader.get_files()
    print(f"Found {len(files)} files. Preparing chunks...")
    
    all_chunks = []
    chunk_map = [] # stores file_path for each chunk
    
    # Token-based chunking
    # We want max ~500 tokens. Leave room for special tokens.
    MAX_TOKENS = 500
    
    for file_path in tqdm(files, desc="Chunking"):
        text = loader.parse_file(file_path)
        if not text:
            continue
        
        # Tokenize (returns {'input_ids': [...]})
        # Use fast tokenizer
        tokenized = tokenizer(text, add_special_tokens=False)
        input_ids = tokenized['input_ids']
        
        # Split into chunks of MAX_TOKENS
        for i in range(0, len(input_ids), MAX_TOKENS):
            chunk_ids = input_ids[i : i + MAX_TOKENS]
            if not chunk_ids: continue
            
            # Decode back to string
            chunk_text = tokenizer.decode(chunk_ids)
            all_chunks.append(chunk_text)
            chunk_map.append(file_path)
            
    print(f"Created {len(all_chunks)} chunks from {len(files)} files.")
    
    indexer = IndexManager()
    
    print("Starting Deep Learning NER (Batched)...")
    
    inference_iter = ner_pipe(all_chunks, batch_size=32)
    
    for params in tqdm(zip(chunk_map, inference_iter), total=len(all_chunks), desc="Inference"):
        file_path, result = params
        
        # Extract entities from this chunk
        chunk_entities = []
        for item in result:
             entity_group = item['entity_group']
             if entity_group in ['PERSON', 'LOCATION', 'ORGANIZATION']:
                short_type = 'PER' if entity_group == 'PERSON' else \
                             'LOC' if entity_group == 'LOCATION' else \
                             'ORG'
                chunk_entities.append((item['word'], short_type))
        
        indexer.add_document(file_path, chunk_entities)
        
    indexer.save('index.pkl')
    
    print("\n--- Top 20 Common Entities (High Accuracy) ---")
    for e, c in indexer.entity_frequency.most_common(20):
        print(f"{e}: {c}")

if __name__ == "__main__":
    main()
