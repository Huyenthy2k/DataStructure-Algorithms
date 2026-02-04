from transformers import pipeline

def test_transformers():
    print("Loading model NlpHUST/ner-vietnamese-electra-base...")
    # aggregation_strategy="simple" merges "B-PER" and "I-PER" into "PER" automatically.
    ner = pipeline("ner", model="NlpHUST/ner-vietnamese-electra-base", aggregation_strategy="simple")
    
    text = "Ái nữ của ông trùm truyền thông Đinh Bá Thành là Đinh Thị Nam Phương, tốt nghiệp Đại học Oxford."
    print(f"Text: {text}")
    
    results = ner(text)
    print("Entities found:")
    for entity in results:
        print(f" - {entity['word']} ({entity['entity_group']}, score: {entity['score']:.4f})")

if __name__ == "__main__":
    test_transformers()
