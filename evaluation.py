import numpy as np
from typing import List, Dict
import pandas as pd
from query_functions import query_handling

def calculate_recall_at_k(relevant_items: List[str], recommended_items: List[str], k: int = 3) -> float:
    """Calculate Recall@K for a single query"""
    if not relevant_items:
        return 0.0
    
    # Get top k recommendations
    top_k = recommended_items[:k]
    
    # Calculate how many relevant items are in top k
    relevant_in_top_k = sum(1 for item in top_k if item in relevant_items)
    
    # Calculate recall
    recall = relevant_in_top_k / len(relevant_items)
    return recall

def calculate_map_at_k(relevant_items: List[str], recommended_items: List[str], k: int = 3) -> float:
    """Calculate MAP@K for a single query"""
    if not relevant_items:
        return 0.0
    
    # Get top k recommendations
    top_k = recommended_items[:k]
    
    # Calculate precision at each position
    precisions = []
    relevant_count = 0
    
    for i, item in enumerate(top_k, 1):
        if item in relevant_items:
            relevant_count += 1
            precision = relevant_count / i
            precisions.append(precision)
    
    # Calculate MAP
    if not precisions:
        return 0.0
    
    return sum(precisions) / len(relevant_items)

def evaluate_recommendations(test_data: pd.DataFrame) -> Dict[str, float]:
    """Evaluate recommendations using Mean Recall@3 and MAP@3"""
    recalls = []
    maps = []
    
    for _, row in test_data.iterrows():
        query = row['query']
        relevant_items = row['relevant_assessments']
        
        # Get recommendations
        recommendations_df = query_handling(query)
        recommended_items = recommendations_df['Assessment Name'].tolist()
        
        # Calculate metrics
        recall = calculate_recall_at_k(relevant_items, recommended_items)
        map_score = calculate_map_at_k(relevant_items, recommended_items)
        
        recalls.append(recall)
        maps.append(map_score)
    
    # Calculate mean scores
    mean_recall = np.mean(recalls)
    mean_map = np.mean(maps)
    
    return {
        "mean_recall_at_3": mean_recall,
        "mean_map_at_3": mean_map
    }

def load_test_data(file_path: str) -> pd.DataFrame:
    """Load and prepare test data"""
    df = pd.read_csv(file_path)
    # Convert string representation of lists to actual lists
    df['relevant_assessments'] = df['relevant_assessments'].apply(eval)
    return df

if __name__ == "__main__":
    # Example usage
    test_data = load_test_data("test_data.csv")
    results = evaluate_recommendations(test_data)
    print("Evaluation Results:")
    print(f"Mean Recall@3: {results['mean_recall_at_3']:.4f}")
    print(f"Mean MAP@3: {results['mean_map_at_3']:.4f}") 