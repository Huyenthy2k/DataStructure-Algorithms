import pickle
from collections import Counter

def inspect():
    try:
        with open('index.pkl', 'rb') as f:
            data = pickle.load(f)
            freq = data.get('entity_frequency', Counter())
            print(f"Total unique entities: {len(freq)}")
            print("Top 20:")
            for e, c in freq.most_common(20):
                print(f"{e}: {c}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect()
