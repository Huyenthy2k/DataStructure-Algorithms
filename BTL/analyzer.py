import os
import json
import pickle
from collections import Counter, defaultdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

try:
    from underthesea import ner
    USING_UNDERTHESEA = True
except ImportError:
    try:
        from pyvi import ViPosTagger, ViTokenizer
        USING_UNDERTHESEA = False
        print("Using pyvi for NER (Approximation via POS tagging).")
    except ImportError:
        USING_UNDERTHESEA = False
        print("Warning: Neither underthesea nor pyvi installed. NER will fail.")

# Standalone helper for multiprocessing
def process_file_wrapper(file_path):
    try:
        # Re-implement parsing here
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            data = json.loads(content)
            text = f"{data.get('Subject', '')} . {data.get('Summary', '')} . {data.get('Content', '')}"
        
        if not text:
            return (file_path, [])

        # Extraction logic
        entities = []
        if USING_UNDERTHESEA:
            try:
                tokens = ner(text)
                current_entity = []
                current_type = None
                for token in tokens:
                    if len(token) == 4: word, _, _, label = token
                    elif len(token) == 3: word, _, label = token
                    else: continue

                    if label.startswith('B-'):
                        if current_entity: entities.append((" ".join(current_entity), current_type))
                        current_entity = [word]
                        current_type = label[2:]
                    elif label.startswith('I-') and current_type == label[2:]:
                        current_entity.append(word)
                    else:
                        if current_entity:
                            entities.append((" ".join(current_entity), current_type))
                            current_entity = []
                            current_type = None
                if current_entity: entities.append((" ".join(current_entity), current_type))
                
                res = [(e[0], e[1]) for e in entities if e[1] in ['PER', 'LOC', 'ORG']]
                return (file_path, res)
            except:
                return (file_path, [])
        else:
            try:
                processed = ViPosTagger.postagging(ViTokenizer.tokenize(text))
                if not processed or len(processed) < 2: return (file_path, [])
                words, tags = processed[0], processed[1]
                for word, tag in zip(words, tags):
                    if tag == 'Np':
                        entities.append((word.replace('_', ' '), 'Np'))
                return (file_path, entities)
            except:
                return (file_path, [])
    except Exception:
        return (file_path, [])

class IndexManager:
    def __init__(self):
        self.inverted_index = defaultdict(set)
        self.entity_frequency = Counter()
        self.co_occurrence = defaultdict(Counter)
    
    def merge_results(self, file_path, entities):
        unique_entities = set([e[0] for e in entities])
        for entity, _ in entities:
             self.inverted_index[entity].add(file_path)
             self.entity_frequency[entity] += 1
             
        unique_list = list(unique_entities)
        for i in range(len(unique_list)):
            e1 = unique_list[i]
            for j in range(i + 1, len(unique_list)):
                e2 = unique_list[j]
                self.co_occurrence[e1][e2] += 1
                self.co_occurrence[e2][e1] += 1

    def save(self, path='index.pkl'):
        # Convert defaultdict to dict for pickling safety if needed, 
        # though defaultdict is pickleable.
        with open(path, 'wb') as f:
            pickle.dump({
                'inverted_index': self.inverted_index,
                'entity_frequency': self.entity_frequency,
                'co_occurrence': self.co_occurrence
            }, f)
        print(f"Index saved to {path} with {len(self.inverted_index)} entities.")

    def load(self, path='index.pkl'):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.inverted_index = data['inverted_index']
            self.entity_frequency = data['entity_frequency']
            self.co_occurrence = data['co_occurrence']
        print(f"Index loaded from {path}")

    def search(self, keyword):
        return self.inverted_index.get(keyword, set())

    def get_top_entities(self, k=20):
        return self.entity_frequency.most_common(k)

    def get_related_entities(self, keyword, k=10):
        return self.co_occurrence.get(keyword, Counter()).most_common(k)

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

def main():
    root_dir = "/Users/nguyensiry/Documents/Code_practice/Data_structure_algorithms/BTL/Article_Crawl"
    loader = DataLoader(root_dir)
    files = loader.get_files()
    print(f"Found {len(files)} files. Starting multiprocessing indexing...")
    
    indexer = IndexManager()
    
    # Use max cores
    num_processes = max(1, cpu_count())
    print(f"Using {num_processes} processes.")
    
    with Pool(processes=num_processes) as pool:
        # imap_unordered for better filling of queue
        results_iter = pool.imap_unordered(process_file_wrapper, files, chunksize=20)
        
        for file_path, entities in tqdm(results_iter, total=len(files), desc="Indexing"):
            indexer.merge_results(file_path, entities)
        
    indexer.save('index.pkl')
    
    print("\n--- Top 20 Common Entities ---")
    for e, c in indexer.get_top_entities(20):
        print(f"{e}: {c}")

if __name__ == "__main__":
    main()
