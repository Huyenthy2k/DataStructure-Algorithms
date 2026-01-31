from analyzer import IndexManager
import sys
import os

def main():
    print("Welcome to Vietnamese Article Named Entity Analysis")
    print("Loading index...")
    
    if not os.path.exists('index.pkl'):
        print("Error: index.pkl not found. Please run analyzer.py first to build the index.")
        sys.exit(1)
        
    manager = IndexManager()
    manager.load('index.pkl')
    
    print(f"Index loaded. {len(manager.inverted_index)} unique entities.")
    
    while True:
        print("\nOptions:")
        print("1. Search articles by entity name")
        print("2. Show top frequent entities")
        print("3. Find related entities (co-occurrence)")
        print("4. Exit")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            keyword = input("Enter entity name to search: ").strip()
            # Normalize? Our extractor keeps original case but maybe we search smartly
            # For now exact match or partial match?
            # Dictionary keys are exact.
            
            results = manager.search(keyword)
            if results:
                print(f"Found {len(results)} articles containing '{keyword}':")
                for i, path in enumerate(list(results)[:10]):
                    print(f" - {os.path.basename(path)}")
                if len(results) > 10:
                    print(f"... and {len(results)-10} more.")
            else:
                # Try case insensitive search
                found = False
                for key in manager.inverted_index:
                    if key.lower() == keyword.lower():
                        results = manager.inverted_index[key]
                        print(f"Did you mean '{key}'? Found {len(results)} articles.")
                        for i, path in enumerate(list(results)[:10]):
                            print(f" - {os.path.basename(path)}")
                        if len(results) > 10:
                            print(f"... and {len(results)-10} more.")
                        found = True
                        break
                if not found:
                    print(f"No articles found for '{keyword}'.")

        elif choice == '2':
            try:
                k = int(input("How many top entities (10-30)? ").strip())
            except:
                k = 20
            print(f"Top {k} entities:")
            for e, c in manager.get_top_entities(k):
                print(f"{e}: {c}")

        elif choice == '3':
            keyword = input("Enter entity name: ").strip()
            related = manager.get_related_entities(keyword, 10)
            if related:
                print(f"Entities related to '{keyword}':")
                for e, c in related:
                    print(f"{e} (co-occurred {c} times)")
            else:
                 # Try case insensitive
                found = False
                for key in manager.co_occurrence:
                    if key.lower() == keyword.lower():
                        related = manager.get_related_entities(key, 10)
                        print(f"Did you mean '{key}'? Related entities:")
                        for e, c in related:
                            print(f"{e} (co-occurred {c} times)")
                        found = True
                        break
                if not found:
                    print(f"No related info for '{keyword}'.")

        elif choice == '4':
            print("Goodbye.")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
