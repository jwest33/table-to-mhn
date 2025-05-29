import pandas as pd
import numpy as np
from hopfield_table import HopfieldTableMemory

def main():
    # Create sample DataFrame with mixed data types
    df = pd.DataFrame({
        'age': [34, 28, 45, 31, 29, 52, 38, 26, 41, 35],
        'salary': [70000, 80000, 60000, 90000, 75000, 120000, 65000, 55000, 95000, 85000],
        'dept': ['sales', 'tech', 'hr', 'tech', 'sales', 'management', 'hr', 'tech', 'sales', 'tech'],
        'experience': [5, 3, 12, 4, 4, 20, 8, 1, 15, 7],
        'location': ['NYC', 'SF', 'NYC', 'SF', 'LA', 'NYC', 'LA', 'SF', 'NYC', 'SF']
    })
    
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*60 + "\n")
    
    # Initialize Hopfield Table Memory
    print("Initializing Hopfield Table Memory...")
    memory = HopfieldTableMemory(df)
    
    # Print memory statistics
    stats = memory.get_memory_stats()
    print("Memory Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("\n" + "="*60 + "\n")
    
    # Example 1: Dense query with all features specified
    print("Example 1: Dense query - Find employees similar to age=30, salary=85000")
    
    # Debug the query first
    memory.debug_query(sparse=False, age=30, salary=85000)
    
    results = memory.query(age=30, salary=85000, top_n=3, visualize=False, sparse=False)
    
    print("Results:")
    for i, res in enumerate(results, 1):
        print(f"\n--- Match {i} ---")
        print(f"Index: {res['index']}")
        print(f"Confidence Score: {res['confidence_score']:.4f}")
        print(f"Distance: {res['distance']:.4f}")
        print("Matched Row:")
        print(res['matched_row'])
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Sparse query with partial information
    print("Example 2: Sparse query - Find tech employees with salary around 80000")
    
    # Debug the sparse query
    memory.debug_query(sparse=True, dept='tech', salary=80000)
    
    results = memory.query(dept='tech', salary=80000, top_n=3, visualize=False, sparse=True)
    
    print("Results:")
    for i, res in enumerate(results, 1):
        print(f"\n--- Match {i} ---")
        print(f"Index: {res['index']}")
        print(f"Confidence Score: {res['confidence_score']:.4f}")
        print(f"Distance: {res['distance']:.4f}")
        print("Matched Row:")
        print(res['matched_row'])
    
    print("\n" + "="*60 + "\n")
    
    # Example 3: Pattern completion
    print("Example 3: Pattern completion - Complete profile for age=33, dept='sales'")
    
    # Debug what we're looking for
    memory.debug_query(sparse=True, age=33, dept='sales')
    
    # Show what a sparse query for these constraints would return
    print("First, let's see what matches these constraints:")
    constraint_matches = memory.query(age=33, dept='sales', top_n=3, sparse=True)
    for i, match in enumerate(constraint_matches, 1):
        print(f"Constraint match {i}: Index {match['index']}, Distance {match['distance']:.4f}")
        print(f"  {match['matched_row'].to_dict()}")
    
    completed = memory.complete_pattern(age=33, dept='sales')
    print("\nCompleted Pattern:")
    print(completed)
    
    print("\n" + "="*60 + "\n")
    
    # Example 4: Query with only categorical information
    print("Example 4: Find employees in HR department")
    results = memory.query(dept='hr', top_n=2, visualize=False, sparse=True)
    
    print("Results:")
    for i, res in enumerate(results, 1):
        print(f"\n--- Match {i} ---")
        print(f"Index: {res['index']}")
        print(f"Confidence Score: {res['confidence_score']:.4f}")
        print(f"Distance: {res['distance']:.4f}")
        print("Matched Row:")
        print(res['matched_row'])
    
    print("\n" + "="*60 + "\n")
    
    # Example 5: Query with only numeric information
    print("Example 5: Find high earners (salary > 100000)")
    results = memory.query(salary=110000, top_n=2, visualize=False, sparse=True)
    
    print("Results:")
    for i, res in enumerate(results, 1):
        print(f"\n--- Match {i} ---")
        print(f"Index: {res['index']}")
        print(f"Confidence Score: {res['confidence_score']:.4f}")
        print(f"Distance: {res['distance']:.4f}")
        print("Matched Row:")
        print(res['matched_row'])
    
    print("\n" + "="*60 + "\n")
    
    # Example 6: Adding new patterns and querying
    print("Example 6: Adding new data and querying")
    new_data = pd.DataFrame({
        'age': [27, 44],
        'salary': [72000, 98000],
        'dept': ['marketing', 'tech'],
        'experience': [3, 12],
        'location': ['LA', 'NYC']
    })
    
    print("Adding new data:")
    print(new_data)
    
    memory.add_patterns(new_data)
    print(f"\nMemory now contains {memory.get_memory_stats()['num_patterns']} patterns")
    
    # Query the expanded memory
    results = memory.query(dept='marketing', top_n=2, visualize=False, sparse=True)
    print("Query for marketing employees:")
    for i, res in enumerate(results, 1):
        print(f"\n--- Match {i} ---")
        print(f"Index: {res['index']}")
        print(f"Confidence Score: {res['confidence_score']:.4f}")
        print("Matched Row:")
        print(res['matched_row'])

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*60)
    print("TESTING EDGE CASES")
    print("="*60 + "\n")
    
    # Edge case 1: DataFrame with only numeric columns
    print("Edge Case 1: Only numeric columns")
    numeric_df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [2, 4, 6, 8, 10],
        'z': [1, 1, 2, 3, 5]
    })
    
    memory_numeric = HopfieldTableMemory(numeric_df)
    results = memory_numeric.query(x=2.5, y=5, top_n=2)
    print("Query results:")
    for res in results:
        print(f"Index {res['index']}: {res['matched_row'].to_dict()}, Distance: {res['distance']:.4f}")
    
    print("\n" + "-"*40 + "\n")
    
    # Edge case 2: DataFrame with only categorical columns
    print("Edge Case 2: Only categorical columns")
    categorical_df = pd.DataFrame({
        'color': ['red', 'blue', 'green', 'red', 'blue'],
        'size': ['S', 'M', 'L', 'M', 'S'],
        'type': ['A', 'B', 'A', 'C', 'B']
    })
    
    memory_categorical = HopfieldTableMemory(categorical_df)
    results = memory_categorical.query(color='red', size='M', top_n=2)
    print("Query results:")
    for res in results:
        print(f"Index {res['index']}: {res['matched_row'].to_dict()}, Distance: {res['distance']:.4f}")
    
    print("\n" + "-"*40 + "\n")
    
    # Edge case 3: Query with unknown categorical values
    print("Edge Case 3: Query with unknown categorical value")
    try:
        results = memory_categorical.query(color='purple', top_n=1, sparse=True)
        print("Query with unknown category handled gracefully:")
        for res in results:
            print(f"Index {res['index']}: {res['matched_row'].to_dict()}")
    except Exception as e:
        print(f"Error handling unknown category: {e}")

if __name__ == "__main__":
    main()
    test_edge_cases()
