import streamlit as st
import pandas as pd
from hopfield_table import HopfieldTableMemory

# Page configuration
st.set_page_config(page_title="Hopfield Table Query", layout="wide")
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .stDataFrame { font-size: 14px; }
    .stSlider > div[data-baseweb="slider"] { margin-top: -10px; }
</style>
""", unsafe_allow_html=True)

st.title("Hopfield Table Query")

# Sample dataset
def load_sample_data():
    return pd.DataFrame({
        'age': [34, 28, 45, 31, 29, 52, 38, 26, 41, 35],
        'salary': [70000, 80000, 60000, 90000, 75000, 120000, 65000, 55000, 95000, 85000],
        'dept': ['sales', 'tech', 'hr', 'tech', 'sales', 'management', 'hr', 'tech', 'sales', 'marketing'],
        'experience': [5, 3, 12, 4, 4, 20, 8, 1, 15, 7],
        'location': ['NYC', 'SF', 'NYC', 'SF', 'LA', 'NYC', 'LA', 'SF', 'NYC', 'SF']
    })

@st.cache_resource
def get_memory():
    return HopfieldTableMemory(load_sample_data())

memory = get_memory()

with st.expander("📊 Memory Statistics", expanded=False):
    stats = memory.get_memory_stats()
    st.write({k: v for k, v in stats.items()})

st.markdown("---")

st.subheader("Query Inputs")
query_inputs = {}

with st.form("query_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input("Age", placeholder="e.g., 30")
        if age.strip(): query_inputs['age'] = int(age.strip())

    with col2:
        salary = st.text_input("Salary", placeholder="e.g., 85000")
        if salary.strip(): query_inputs['salary'] = int(salary.strip())

    with col3:
        dept = st.selectbox("Department", ["", "sales", "tech", "hr", "management", "marketing"])
        if dept: query_inputs['dept'] = dept

    col4, col5 = st.columns(2)
    with col4:
        experience = st.text_input("Experience (years)", placeholder="e.g., 5")
        if experience.strip(): query_inputs['experience'] = int(experience.strip())

    with col5:
        location = st.selectbox("Location", ["", "NYC", "SF", "LA"])
        if location: query_inputs['location'] = location

    col6, col7 = st.columns([3, 2])
    with col6:
        top_n = st.slider("Top Matches", 1, 10, 3)
    with col7:
        sparse = st.checkbox("Sparse Query", value=True)

    submitted = st.form_submit_button("Run Query")

if submitted:
    with st.spinner("Searching for similar entries..."):
        results = memory.query(top_n=top_n, sparse=sparse, **query_inputs)

    if results:
        st.success(f"Found {len(results)} matches")
        for i, res in enumerate(results, 1):
            with st.expander(f"Match {i}  |  Confidence: {res['confidence_score']:.3f}  |  Distance: {res['distance']:.3f}"):
                st.dataframe(pd.DataFrame([res['matched_row']]))
    else:
        st.warning("No results found. Try adjusting your parameters.")

with st.expander("Debug Info"):
    st.json(query_inputs)
    memory.debug_query(sparse=sparse, **query_inputs)
