import streamlit as st
import pandas as pd
import numpy as np
from hopfield_table import HopfieldTableMemory
from typing import Dict, Any, Optional, List
import io
import os

# Page configuration
st.set_page_config(page_title="Hopfield Table Query", layout="wide")
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .stDataFrame { font-size: 14px; }
    .stSlider > div[data-baseweb="slider"] { margin-top: -10px; }
    .upload-section { 
        border: 2px dashed #cccccc; 
        border-radius: 10px; 
        padding: 20px; 
        margin: 10px 0; 
        text-align: center;
    }
    .data-info {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("Hopfield Table Query System")
st.markdown("Upload your CSV data and perform intelligent similarity searches using Hopfield networks.")

def load_default_data():
    """Load default employees.csv dataset if available."""
    if os.path.exists('employees.csv'):
        try:
            df = pd.read_csv('employees.csv')
            return df, "employees.csv (default)"
        except Exception as e:
            st.warning(f"Error loading employees.csv: {str(e)}. Using fallback sample data.")
            return load_sample_data(), "Sample data (fallback)"
    else:
        st.info("employees.csv not found. Using sample data. Upload a CSV or generate employees.csv using the data generator.")
        return load_sample_data(), "Sample data (employees.csv not found)"

# Sample dataset (fallback)
def load_sample_data():
    """Load sample dataset as fallback when employees.csv is not available."""
    return pd.DataFrame({
        'age': [34, 28, 45, 31, 29, 52, 38, 26, 41, 35],
        'salary': [70000, 80000, 60000, 90000, 75000, 120000, 65000, 55000, 95000, 85000],
        'dept': ['sales', 'tech', 'hr', 'tech', 'sales', 'management', 'hr', 'tech', 'sales', 'marketing'],
        'experience': [5, 3, 12, 4, 4, 20, 8, 1, 15, 7],
        'location': ['NYC', 'SF', 'NYC', 'SF', 'LA', 'NYC', 'LA', 'SF', 'NYC', 'SF']
    })

def load_and_validate_csv(uploaded_file) -> Optional[pd.DataFrame]:
    """Load and validate uploaded CSV file."""
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Basic validation
        if df.empty:
            st.error("The uploaded CSV file is empty.")
            return None
        
        if len(df.columns) < 2:
            st.error("CSV must have at least 2 columns for meaningful similarity search.")
            return None
        
        # Check for excessive missing values
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        problematic_cols = missing_percentage[missing_percentage > 50].index.tolist()
        
        if problematic_cols:
            st.warning(f"Columns with >50% missing values: {problematic_cols}. Consider removing these columns.")
        
        # Basic data type inference and cleanup
        for col in df.columns:
            # Try to convert string numbers to numeric
            if df[col].dtype == 'object':
                # Try to convert to numeric if it looks like numbers
                numeric_version = pd.to_numeric(df[col], errors='coerce')
                if not numeric_version.isna().all() and numeric_version.notna().sum() / len(df) > 0.8:
                    df[col] = numeric_version
        
        return df
        
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None

def get_column_info(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Analyze DataFrame columns and return metadata for form generation."""
    column_info = {}
    
    for col in df.columns:
        info = {
            'dtype': str(df[col].dtype),
            'unique_count': df[col].nunique(),
            'null_count': df[col].isnull().sum(),
            'is_numeric': pd.api.types.is_numeric_dtype(df[col]),
            'is_categorical': pd.api.types.is_object_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype)
        }
        
        if info['is_numeric']:
            info['min'] = df[col].min()
            info['max'] = df[col].max()
            info['mean'] = df[col].mean()
        
        if info['is_categorical'] and info['unique_count'] <= 50:  # Reasonable limit for selectbox
            info['unique_values'] = sorted(df[col].dropna().unique().tolist())
        
        column_info[col] = info
    
    return column_info

@st.cache_data
def get_cached_csv_info(csv_content: bytes) -> tuple:
    """Cache CSV processing to avoid reloading on every interaction."""
    df = pd.read_csv(io.BytesIO(csv_content))
    # Apply the same validation logic as load_and_validate_csv
    for col in df.columns:
        if df[col].dtype == 'object':
            numeric_version = pd.to_numeric(df[col], errors='coerce')
            if not numeric_version.isna().all() and numeric_version.notna().sum() / len(df) > 0.8:
                df[col] = numeric_version
    
    column_info = get_column_info(df)
    return df, column_info

@st.cache_resource
def get_memory(_df: pd.DataFrame) -> HopfieldTableMemory:
    """Create and cache Hopfield memory. Use _df to avoid hashing issues."""
    return HopfieldTableMemory(_df)

def create_dynamic_form(df: pd.DataFrame, column_info: Dict[str, Dict[str, Any]]) -> tuple:
    """Create a dynamic form based on the DataFrame columns."""
    st.subheader("Query Parameters")
    st.markdown("*Fill in the fields you want to match. Leave others empty for flexible matching.*")
    
    # Organize columns for better layout
    numeric_cols = [col for col, info in column_info.items() if info['is_numeric']]
    categorical_cols = [col for col, info in column_info.items() if info['is_categorical']]
    
    # Create form
    with st.form("dynamic_query_form"):
        query_inputs = {}
        
        # Numeric columns
        if numeric_cols:
            st.markdown("**Numeric Fields**")
            num_cols = st.columns(min(3, len(numeric_cols)))
            
            for i, col in enumerate(numeric_cols):
                with num_cols[i % len(num_cols)]:
                    info = column_info[col]
                    
                    # Create appropriate input based on data range
                    # Handle boolean or small range numeric data
                    try:
                        range_size = info['max'] - info['min'] if pd.api.types.is_numeric_dtype(type(info['max'])) else 0
                        use_selectbox = info['unique_count'] <= 20 and range_size <= 100
                    except (TypeError, ValueError):
                        # Handle cases where max/min are not numeric (e.g., boolean, datetime)
                        use_selectbox = info['unique_count'] <= 20
                    
                    if use_selectbox:
                        # Use selectbox for small ranges
                        unique_vals = sorted(df[col].dropna().unique())
                        selected = st.selectbox(
                            f"{col.title()}", 
                            options=[""] + [str(v) for v in unique_vals],
                            key=f"num_{col}"
                        )
                        if selected:
                            query_inputs[col] = type(unique_vals[0])(selected)
                    else:
                        # Use number input for larger ranges
                        try:
                            value = st.number_input(
                                f"{col.title()}", 
                                value=None,
                                placeholder=f"e.g., {info['mean']:.0f}",
                                key=f"num_{col}"
                            )
                            if value is not None:
                                query_inputs[col] = value
                        except (ValueError, TypeError):
                            # Fallback to text input for problematic numeric columns
                            value = st.text_input(
                                f"{col.title()}", 
                                placeholder=f"Enter {col}...",
                                key=f"num_{col}_fallback"
                            )
                            if value.strip():
                                try:
                                    # Try to convert to appropriate type
                                    query_inputs[col] = pd.to_numeric(value.strip())
                                except:
                                    query_inputs[col] = value.strip()
        
        # Categorical columns
        if categorical_cols:
            st.markdown("**Categorical Fields**")
            cat_cols = st.columns(min(3, len(categorical_cols)))
            
            for i, col in enumerate(categorical_cols):
                with cat_cols[i % len(cat_cols)]:
                    info = column_info[col]
                    
                    if 'unique_values' in info:
                        # Use selectbox for reasonable number of categories
                        selected = st.selectbox(
                            f"{col.title()}", 
                            options=[""] + info['unique_values'],
                            key=f"cat_{col}"
                        )
                        if selected:
                            query_inputs[col] = selected
                    else:
                        # Use text input for high cardinality
                        value = st.text_input(
                            f"{col.title()}", 
                            placeholder=f"Enter {col}...",
                            key=f"cat_{col}"
                        )
                        if value.strip():
                            query_inputs[col] = value.strip()
        
        # Query options
        st.markdown("**Search Options**")
        opt_col1, opt_col2, opt_col3 = st.columns(3)
        
        with opt_col1:
            top_n = st.slider("Top Matches", 1, min(20, len(df)), 5)
        
        with opt_col2:
            sparse = st.checkbox("Sparse Query", value=True, 
                               help="Only match specified fields vs. all fields")
        
        with opt_col3:
            show_debug = st.checkbox("Show Debug Info", value=False)
        
        # Submit button
        submitted = st.form_submit_button("Search Similar Records", use_container_width=True)
        
        return query_inputs, top_n, sparse, show_debug, submitted

def display_results(results: List[Dict[str, Any]], df: pd.DataFrame) -> None:
    """Display search results in a single comprehensive dataframe."""
    if not results:
        st.warning("No results found. Try adjusting your parameters or using fewer constraints.")
        return
    
    st.success(f"Found {len(results)} matches")
    
    # Create comprehensive results dataframe
    results_data = []
    for i, res in enumerate(results, 1):
        row_data = res['matched_row'].copy()
        # Add match metadata as the first columns
        row_data = {
            'Match_Rank': i,
            'Confidence': round(res['confidence_score'], 4),
            'Distance': round(res['distance'], 4),
            'Original_Index': res['index'],
            **row_data  # Add all original columns
        }
        results_data.append(row_data)
    
    results_df = pd.DataFrame(results_data)
    
    # Display the results dataframe
    st.dataframe(results_df, use_container_width=True)
    
    # Optional: Add a download button for the results
    csv_data = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv_data,
        file_name="hopfield_search_results.csv",
        mime="text/csv"
    )

# Main application logic
def main():
    # File upload section
    st.markdown("## Data Upload")
    
    with st.container():
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your CSV dataset for similarity search. The system will automatically detect column types and create appropriate query inputs."
        )
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if uploaded_file is not None:
                st.success(f"File uploaded: **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")
            else:
                st.info("No file uploaded. Loading default employees.csv dataset.")
        
        with col2:
            use_sample = st.button("Use Sample Data", help="Load sample employee data")
    
    # Load and process data
    if uploaded_file is not None:
        # Use uploaded CSV
        csv_content = uploaded_file.read()
        uploaded_file.seek(0)  # Reset file pointer
        
        try:
            df, column_info = get_cached_csv_info(csv_content)
            data_source = f"Uploaded: {uploaded_file.name}"
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
            return
            
    elif use_sample or 'df' not in st.session_state:
        # Load default data (employees.csv or sample)
        df, data_source = load_default_data()
        column_info = get_column_info(df)
    else:
        # Use previously loaded data
        df = st.session_state.df
        column_info = st.session_state.column_info
        data_source = st.session_state.get('data_source', 'Unknown')
    
    # Store in session state
    st.session_state.df = df
    st.session_state.column_info = column_info
    st.session_state.data_source = data_source
    
    # Dataset information
    st.markdown("## Dataset Information")
    
    with st.expander("Dataset Overview", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("**Records**", f"{len(df):,}")
        with col2:
            st.metric("**Columns**", len(df.columns))
        with col3:
            numeric_count = sum(1 for info in column_info.values() if info['is_numeric'])
            st.metric("**Numeric**", numeric_count)
        with col4:
            categorical_count = sum(1 for info in column_info.values() if info['is_categorical'])
            st.metric("**Categorical**", categorical_count)
        
        st.markdown(f"**Data Source:** {data_source}")
        
        # Column details
        st.markdown("**Column Details:**")
        col_details = []
        for col, info in column_info.items():
            col_type = "Numeric" if info['is_numeric'] else "Categorical"
            missing_pct = (info['null_count'] / len(df)) * 100
            col_details.append({
                'Column': col,
                'Type': col_type,
                'Unique Values': info['unique_count'],
                'Missing (%)': f"{missing_pct:.1f}%"
            })
        
        st.dataframe(pd.DataFrame(col_details), use_container_width=True)
        
        # Sample data preview
        st.markdown("**Sample Data:**")
        st.dataframe(df.head(), use_container_width=True)
    
    # Create Hopfield memory
    try:
        memory = get_memory(df)
        
        # Memory statistics
        with st.expander("Hopfield Memory Statistics", expanded=False):
            stats = memory.get_memory_stats()
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            
            with stat_col1:
                st.metric("**Stored Patterns**", stats['num_patterns'])
                st.metric("**Pattern Dimension**", stats['pattern_dimension'])
            
            with stat_col2:
                st.metric("**Numeric Features**", stats['numeric_features'])
                st.metric("**Categorical Features**", stats['categorical_features'])
            
            with stat_col3:
                st.metric("**Total Cat Dimensions**", stats['total_categorical_dimensions'])
                status = "Available" if stats['hopfield_available'] else "Fallback Mode"
                st.markdown(f"**Hopfield Status:** {status}")
    
    except Exception as e:
        st.error(f"Error creating Hopfield memory: {str(e)}")
        return
    
    st.markdown("---")
    
    # Dynamic query form
    query_inputs, top_n, sparse, show_debug, submitted = create_dynamic_form(df, column_info)
    
    # Execute query
    if submitted:
        if not query_inputs:
            st.warning("Please specify at least one query parameter.")
        else:
            with st.spinner("Searching for similar records..."):
                try:
                    results = memory.query(top_n=top_n, sparse=sparse, **query_inputs)
                    display_results(results, df)
                    
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
    
    # Debug information
    if show_debug and query_inputs:
        st.markdown("---")
        with st.expander("Debug Information", expanded=True):
            st.json({
                "query_inputs": query_inputs,
                "sparse_mode": sparse,
                "top_n": top_n,
                "dataset_shape": df.shape
            })
            
            # Show debug output from memory
            st.text("Hopfield Memory Debug Output:")
            memory.debug_query(sparse=sparse, **query_inputs)

if __name__ == "__main__":
    main()
