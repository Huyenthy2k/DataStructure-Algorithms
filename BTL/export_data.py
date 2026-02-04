import pickle
import pandas as pd
import os

def main():
    index_path = 'index.pkl'
    if not os.path.exists(index_path):
        print(f"Error: {index_path} not found. Please wait for analyzer.py to finish.")
        return

    print(f"Loading {index_path}...")
    with open(index_path, 'rb') as f:
        data = pickle.load(f)
    
    # Export Frequency Report
    print("Exporting entities_report.csv...")
    freqs = data['entity_frequency']
    df_freq = pd.DataFrame(freqs.most_common(), columns=['Entity', 'Frequency'])
    # Add Rank (1-based)
    df_freq.index = df_freq.index + 1
    df_freq.to_csv('entities_report.csv', index_label='Rank')
    print(f"Saved {len(df_freq)} entities to entities_report.csv")

    # Export Co-occurrence (Edges) for Gephi/Network analysis
    # Format: Source, Target, Weight
    print("Exporting co_occurrence.csv (Top 1000 edges)...")
    co_occurrence = data['co_occurrence']
    edges = []
    
    # We only take significant edges to avoid massive file size
    # Let's filter for edges with weight > 1 or just take top N
    
    # Flatten dict of dicts
    for source, targets in co_occurrence.items():
        for target, weight in targets.items():
            if source < target: # Avoid duplicates for undirected graph
                edges.append((source, target, weight))
    
    # Sort by weight
    edges.sort(key=lambda x: x[2], reverse=True)
    
    # Take top 5000 edges
    top_edges = edges[:5000]
    df_edges = pd.DataFrame(top_edges, columns=['Source', 'Target', 'Weight'])
    df_edges.to_csv('co_occurrence.csv', index=False)
    print(f"Saved {len(df_edges)} co-occurrence edges to co_occurrence.csv")

if __name__ == "__main__":
    main()
