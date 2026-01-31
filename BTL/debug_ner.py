import json
from pyvi import ViPosTagger
import os

def debug_one_file():
    path = "/Users/nguyensiry/Documents/Code_practice/Data_structure_algorithms/BTL/Article_Crawl/afamily_vn/2025-05-19_afamily.vn_11-52-13.txt"
    try:
        # Test hardcoded
        from pyvi import ViPosTagger, ViTokenizer
        print("Testing hardcoded string:")
        test_str = "Hà Nội là thủ đô của Việt Nam"
        print(ViPosTagger.postagging(ViTokenizer.tokenize(test_str)))
        
        path = "/Users/nguyensiry/Documents/Code_practice/Data_structure_algorithms/BTL/Article_Crawl/afamily_vn/2025-05-19_afamily.vn_11-52-13.txt"
        with open(path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            data = json.loads(content)
            text_content = f"{data.get('Subject', '')} . {data.get('Summary', '')} . {data.get('Content', '')}"
            print(f"Text Content Length: {len(text_content)}")
            print(f"Sample Text: {text_content[:200]}")
            
            processed = ViPosTagger.postag(text_content)
            print(f"Processed Type: {type(processed)}")
            if processed:
                print(f"Structure: {len(processed)} items")
                if len(processed) >= 2:
                    words = processed[0]
                    tags = processed[1]
                    print(f"First 20 words: {words[:20]}")
                    print(f"First 20 tags: {tags[:20]}")
                    
                    found_np = False
                    for w, t in zip(words, tags):
                        if t == 'Np':
                            print(f"Found Np: {w}")
                            found_np = True
                            break
                    if not found_np:
                        print("No Np tags found in sample.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_one_file()
