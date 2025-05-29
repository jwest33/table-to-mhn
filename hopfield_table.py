import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from hflayers import Hopfield
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Any, Optional, Union


class HopfieldTableMemory:
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize Hopfield-based table memory for similarity search.
        
        Args:
            dataframe: Input pandas DataFrame to store as patterns
        """
        self.df = dataframe.copy()
        
        # Separate numeric and categorical columns
        self.numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Compute defaults for missing values
        if self.numeric_cols:
            self.numeric_defaults = self.df[self.numeric_cols].median().to_dict()
        else:
            self.numeric_defaults = {}
            
        if self.categorical_cols:
            self.categorical_defaults = (
                self.df[self.categorical_cols]
                .apply(lambda s: Counter(s).most_common(1)[0][0])
                .to_dict()
            )
        else:
            self.categorical_defaults = {}
        
        # Initialize and fit preprocessors
        self._setup_preprocessors()
        
        # Create encoded patterns and setup Hopfield network
        self._create_patterns()
        self._setup_hopfield_network()
    
    def _setup_preprocessors(self) -> None:
        """Setup and fit the data preprocessors."""
        self.scaler = MinMaxScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        
        # Fit numeric scaler
        if self.numeric_cols:
            self.scaler.fit(self.df[self.numeric_cols])
            
        # Fit categorical encoder
        if self.categorical_cols:
            self.encoder.fit(self.df[self.categorical_cols])
    
    def _create_patterns(self) -> None:
        """Create encoded patterns from the DataFrame."""
        n_rows = len(self.df)
        
        # Encode numeric columns
        if self.numeric_cols:
            self.encoded_numeric = self.scaler.transform(self.df[self.numeric_cols])
        else:
            self.encoded_numeric = np.zeros((n_rows, 0))
        
        # Encode categorical columns
        if self.categorical_cols:
            self.encoded_categorical = self.encoder.transform(self.df[self.categorical_cols]).astype(float)
        else:
            self.encoded_categorical = np.zeros((n_rows, 0))
        
        # Combine all features
        self.full_patterns = np.hstack([self.encoded_numeric, self.encoded_categorical])
        self.pattern_dim = self.full_patterns.shape[1]
        
        # Convert to PyTorch tensors with proper shape for Hopfield layer
        # Shape: (sequence_length=n_rows, batch_size=1, feature_dim)
        self.stored_patterns = torch.tensor(
            self.full_patterns, dtype=torch.float32
        ).unsqueeze(1)  # Add batch dimension
    
    def _setup_hopfield_network(self) -> None:
        """Initialize the Hopfield network."""
        try:
            self.hopfield = Hopfield(input_size=self.pattern_dim)
            self.hopfield_available = True
        except Exception as e:
            print(f"Warning: Hopfield initialization failed: {e}")
            print("Falling back to direct similarity computation")
            self.hopfield = None
            self.hopfield_available = False
    
    def _encode_query(self, **kwargs) -> torch.Tensor:
        """
        Encode query parameters into the same feature space as stored patterns.
        
        Args:
            **kwargs: Query parameters (column_name=value pairs)
            
        Returns:
            Encoded query tensor with shape (1, 1, feature_dim)
        """
        # Handle numeric features
        if self.numeric_cols:
            num_values = {}
            for col in self.numeric_cols:
                if col in kwargs:
                    num_values[col] = kwargs[col]
                else:
                    num_values[col] = self.numeric_defaults.get(col, 0.0)
            
            # Create DataFrame with correct column order
            num_df = pd.DataFrame([num_values])[self.numeric_cols]  # Ensure column order
            num_encoded = self.scaler.transform(num_df)
        else:
            num_encoded = np.zeros((1, 0))
        
        # Handle categorical features
        if self.categorical_cols:
            cat_values = {}
            for col in self.categorical_cols:
                if col in kwargs:
                    cat_values[col] = kwargs[col]
                else:
                    cat_values[col] = self.categorical_defaults.get(col, "")
            
            # Create DataFrame with correct column order
            cat_df = pd.DataFrame([cat_values])[self.categorical_cols]  # Ensure column order
            cat_encoded = self.encoder.transform(cat_df)
        else:
            cat_encoded = np.zeros((1, 0))
        
        # Combine and convert to tensor
        query_pattern = np.hstack([num_encoded, cat_encoded])
        return torch.tensor(query_pattern, dtype=torch.float32).unsqueeze(1)
    
    def _create_sparse_query(self, **kwargs) -> torch.Tensor:
        """
        Create a sparse query where specified features are used and unspecified ones are masked.
        
        Args:
            **kwargs: Query parameters (column_name=value pairs)
            
        Returns:
            Tuple of (query_tensor, mask_tensor) where mask indicates which features are specified
        """
        # Start with zeros
        query_pattern = np.zeros((1, self.pattern_dim))
        mask = np.zeros((1, self.pattern_dim), dtype=bool)
        
        feature_idx = 0
        
        # Handle numeric columns
        if self.numeric_cols:
            # Create a complete DataFrame with default values
            num_values = {col: self.numeric_defaults[col] for col in self.numeric_cols}
            
            # Update with provided values and mark which ones were specified
            specified_numeric = []
            for col in self.numeric_cols:
                if col in kwargs:
                    num_values[col] = kwargs[col]
                    specified_numeric.append(col)
            
            # Transform the complete DataFrame
            temp_df = pd.DataFrame([num_values])[self.numeric_cols]
            scaled_values = self.scaler.transform(temp_df)[0]
            
            # Set all scaled values in the query pattern
            query_pattern[0, feature_idx:feature_idx + len(self.numeric_cols)] = scaled_values
            
            # Set mask only for specified columns
            for i, col in enumerate(self.numeric_cols):
                if col in specified_numeric:
                    mask[0, feature_idx + i] = True
            
            feature_idx += len(self.numeric_cols)
        
        # Handle categorical columns
        if self.categorical_cols:
            # Create complete DataFrame with default values
            cat_values = {col: self.categorical_defaults[col] for col in self.categorical_cols}
            
            # Update with provided values and track specified ones
            specified_categorical = []
            for col in self.categorical_cols:
                if col in kwargs:
                    cat_values[col] = kwargs[col]
                    specified_categorical.append(col)
            
            # Transform the complete DataFrame
            temp_df = pd.DataFrame([cat_values])[self.categorical_cols]
            encoded_values = self.encoder.transform(temp_df)[0]
            
            # Set all encoded values in the query pattern
            query_pattern[0, feature_idx:feature_idx + len(encoded_values)] = encoded_values
            
            # Set mask only for specified categorical columns
            cat_start_idx = 0
            for col in self.categorical_cols:
                col_idx = self.categorical_cols.index(col)
                col_length = len(self.encoder.categories_[col_idx])
                
                if col in specified_categorical:
                    mask[0, feature_idx + cat_start_idx:feature_idx + cat_start_idx + col_length] = True
                
                cat_start_idx += col_length
        
        query_tensor = torch.tensor(query_pattern, dtype=torch.float32).unsqueeze(1)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(1)
        
        return query_tensor, mask_tensor
    
    def query(self, top_n: int = 1, visualize: bool = False, sparse: bool = True, **kwargs) -> List[Dict[str, Any]]:
        """
        Query the Hopfield memory for similar patterns.
        
        Args:
            top_n: Number of top matches to return
            visualize: Whether to create a visualization
            sparse: Whether to use sparse querying (only match specified features)
            **kwargs: Query parameters (column_name=value pairs)
            
        Returns:
            List of dictionaries containing match information
        """
        with torch.no_grad():
            if sparse and len(kwargs) > 0 and len(kwargs) < len(self.df.columns):
                # Sparse query: only match on specified features
                query_tensor, mask_tensor = self._create_sparse_query(**kwargs)
                
                # Compute distances only on masked (specified) features
                # Only consider the dimensions where mask is True
                mask_bool = mask_tensor.squeeze().bool()
                
                if mask_bool.any():
                    # Extract only the masked features for comparison
                    query_masked = query_tensor.squeeze()[mask_bool]
                    patterns_masked = self.stored_patterns.squeeze(1)[:, mask_bool]
                    
                    # Compute distances only on specified features
                    distances = torch.norm(patterns_masked - query_masked.unsqueeze(0), dim=1)
                else:
                    # No features specified, use all patterns with equal distance
                    distances = torch.zeros(len(self.df))
            else:
                # Dense query: match on all features
                query_tensor = self._encode_query(**kwargs)
                
                # Compute distances from query to all stored patterns
                distances = torch.norm(
                    self.stored_patterns.squeeze(1) - query_tensor.squeeze(1), 
                    dim=1
                )
        
        # Get top-k closest matches
        if len(distances) > 0:
            _, closest_indices = torch.topk(-distances, k=min(top_n, len(self.df)))
        else:
            closest_indices = torch.arange(min(top_n, len(self.df)))
        
        if visualize and len(distances) > 0:
            self._visualize_distances(distances, closest_indices)
        
        # Prepare results
        results = []
        for idx in closest_indices:
            distance = distances[idx].item() if len(distances) > 0 else 0.0
            confidence = 1.0 / (1.0 + distance)
            
            results.append({
                "index": idx.item(),
                "matched_row": self.df.iloc[idx.item()].copy(),
                "confidence_score": confidence,
                "distance": distance
            })
        
        return results
    
    def _visualize_distances(self, distances: torch.Tensor, highlight_indices: torch.Tensor) -> None:
        """Visualize the distance distribution with highlighted matches."""
        distances_np = distances.detach().cpu().numpy()
        
        plt.figure(figsize=(10, 6))
        plt.title("Hopfield Memory - Pattern Distances")
        
        # Plot all distances
        bars = plt.bar(range(len(distances_np)), distances_np, alpha=0.6, color='lightblue')
        
        # Highlight selected matches
        for idx in highlight_indices:
            bars[idx].set_color('red')
            bars[idx].set_alpha(0.9)
        
        plt.xlabel("Pattern Index")
        plt.ylabel("Distance from Query")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def add_patterns(self, new_df: pd.DataFrame) -> None:
        """
        Add new patterns to the memory (requires retraining).
        
        Args:
            new_df: New DataFrame to add to the memory
        """
        # Combine with existing data
        self.df = pd.concat([self.df, new_df], ignore_index=True)
        
        # Recreate patterns and network
        self._create_patterns()
        self._setup_hopfield_network()
    
    def debug_query(self, sparse: bool = True, **kwargs) -> None:
        """Debug method to understand query behavior."""
        print(f"\nDEBUG: Query parameters: {kwargs}")
        print(f"DEBUG: Sparse mode: {sparse}")
        
        # Show the encoded query
        query_tensor = self._encode_query(**kwargs)
        print(f"Encoded query shape: {query_tensor.shape}")
        print(f"Encoded query values: {query_tensor.squeeze().numpy()}")
        
        # Show stored patterns for comparison
        print(f"Stored patterns shape: {self.stored_patterns.shape}")
        print("First few stored patterns:")
        for i in range(min(3, len(self.df))):
            print(f"  Pattern {i}: {self.stored_patterns[i].squeeze().numpy()}")
            print(f"  Original row {i}: {self.df.iloc[i].to_dict()}")
        
        # Test sparse query only if in sparse mode
        if sparse and len(kwargs) > 0 and len(kwargs) < len(self.df.columns):
            query_sparse, mask = self._create_sparse_query(**kwargs)
            print(f"Sparse query: {query_sparse.squeeze().numpy()}")
            print(f"Mask: {mask.squeeze().numpy()}")
            print(f"Masked features count: {mask.sum().item()}")
        else:
            print("Using DENSE query - comparing all features")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory."""
        return {
            "num_patterns": len(self.df),
            "pattern_dimension": self.pattern_dim,
            "numeric_features": len(self.numeric_cols),
            "categorical_features": len(self.categorical_cols),
            "total_categorical_dimensions": sum(len(cats) for cats in self.encoder.categories_) if self.categorical_cols else 0,
            "hopfield_available": self.hopfield_available
        }
