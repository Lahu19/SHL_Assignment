import streamlit as st
import pandas as pd
from query_functions import query_handling
import html
import re

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("combined_assessment.csv")
        if df.empty:
            st.error("No data found in the CSV file!")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading assessment data: {str(e)}")
        return pd.DataFrame()
    
    # Ensure all required columns exist
    required_columns = ['Assessment Name', 'Test Type', 'Remote Testing', 'Adaptive/IRT', 'Relative URL']
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
    
    # Clean duration data
    df['Duration in mins'] = None
    if 'Assessment Length' in df.columns:
        df['Duration in mins'] = df['Assessment Length'].apply(
            lambda x: float(re.search(r'\d+', str(x)).group()) if pd.notna(x) and re.search(r'\d+', str(x)) else None
        )
    
    return df

# Page configuration
st.set_page_config(
    page_title="SHL Assessment Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with updated button styling
st.markdown("""
    <style>
        .stcard {
            background-color: #1e1e1e;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #333;
            color: #ffffff;
        }
        
        .card-title {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #ffffff;
        }
        
        .tag-container {
            margin: 10px 0;
        }
        
        .tag {
            background-color: rgba(75, 139, 190, 0.2);
            color: #4B8BBE;
            padding: 5px 12px;
            border-radius: 15px;
            margin-right: 8px;
            margin-bottom: 8px;
            display: inline-block;
            font-size: 12px;
        }
        
        .card-description {
            color: #cccccc;
            margin: 15px 0;
            line-height: 1.5;
            font-size: 14px;
        }
        
        .card-duration {
            color: #888888;
            margin: 10px 0;
            font-size: 14px;
        }
        
        .card-footer {
            margin-top: 15px;
        }
        
        .card-link {
            background-color: #4B8BBE;
            color: #000000 !important;
            padding: 8px 16px;
            border-radius: 5px;
            text-decoration: none !important;
            display: inline-block;
            transition: background-color 0.3s;
            font-weight: 600;
        }
        
        .card-link:hover {
            background-color: #3a7a9d;
            color: #000000 !important;
            text-decoration: none !important;
        }
        
        /* Search box styling */
        .stTextInput input {
            border-radius: 25px;
            border: 2px solid #4B8BBE;
            padding: 10px 20px;
        }
        
        /* Button styling */
        .stButton button {
            background: linear-gradient(45deg, #4B8BBE, #306998);
            color: #000000;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: 600;
        }
        
        /* Header styling */
        h1 {
            color: #4B8BBE;
            text-align: center;
            margin-bottom: 30px;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar filters
with st.sidebar:
    st.markdown("## 🎯 Filters")
    
    # Sort options
    sort_by = st.selectbox(
        "Sort By",
        ["Assessment Name", "Test Type", "Duration in mins"],
        index=0
    )
    
    sort_order = st.radio(
        "Sort Order",
        ["Ascending", "Descending"],
        horizontal=True
    )
    
    test_type_filter = st.multiselect(
        "Test Type",
        ["Knowledge & Skills", "Personality & Behavior", "Ability & Aptitude", "Simulations"],
        default=None
    )
    
    duration_filter = st.slider(
        "Max Duration (minutes)",
        min_value=0,
        max_value=60,
        value=60
    )
    
    remote_filter = st.checkbox("Remote Testing Only", value=False)
    adaptive_filter = st.checkbox("Adaptive Tests Only", value=False)

# Main content
st.markdown("# 🧠 SHL Assessment Recommandation System")
st.markdown(
    "<p style='text-align: center; color: #ccc; font-size: 18px; margin-bottom: 30px;'>"
    "AI-powered assessment recommandation system"
    "</p>",
    unsafe_allow_html=True
)

# Search box with example queries
example_queries = [
    "Python programming test",
    "Leadership assessment",
    "Data analysis skills",
    "Project management certification"
]
query = st.text_input(
    "🔍 Search for specific assessments (optional)",
    placeholder=f"Try: {' | '.join(example_queries)}"
)

def clean_text(text):
    """Clean and format text content."""
    if pd.isna(text) or not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', str(text))
    # Fix common HTML entities
    text = html.unescape(text)  # This will handle all HTML entities properly
    return text.strip()

def format_url(url, name):
    """Format URL with proper domain."""
    if not url or pd.isna(url):
        sanitized_name = name.lower().replace(' ', '-').replace('(', '').replace(')', '')
        return f"https://www.shl.com/solutions/products/product-catalog/view/{sanitized_name}/"
    elif not url.startswith(('http://', 'https://')):
        # Remove any leading slashes and ensure proper formatting
        url = url.lstrip('/')
        return f"https://www.shl.com/{url}"
    return url

def prepare_table_data(df):
    """Prepare dataframe for table display."""
    if df.empty:
        return pd.DataFrame()
        
    # Clean and prepare data
    display_df = df.copy()
    
    # Clean text in all columns
    for col in display_df.columns:
        display_df[col] = display_df[col].apply(clean_text)
    
    # Format duration
    if 'Duration in mins' in display_df.columns:
        display_df['Duration'] = display_df['Duration in mins'].apply(
            lambda x: f"{int(float(x))} minutes" if pd.notna(x) and str(x).strip() else "Not specified"
        )
    else:
        display_df['Duration'] = "Not specified"
    
    # Format Remote Testing and Adaptive columns
    if 'Remote Testing' in display_df.columns:
        display_df['Remote Testing'] = display_df['Remote Testing'].apply(
            lambda x: '✓' if str(x).lower() == 'yes' else '✗'
        )
    else:
        display_df['Remote Testing'] = '✗'
        
    if 'Adaptive/IRT' in display_df.columns:
        display_df['Adaptive'] = display_df['Adaptive/IRT'].apply(
            lambda x: '✓' if str(x).lower() == 'yes' else '✗'
        )
    else:
        display_df['Adaptive'] = '✗'
    
    # Format URLs
    display_df['Relative URL'] = display_df.apply(
        lambda row: format_url(row['Relative URL'], row['Assessment Name']), 
        axis=1
    )
    
    return display_df

# Display results
if query:
    results_df = query_handling(query)
    if not results_df.empty:
        st.markdown("### 🔍 Search Results")
        for _, row in results_df.iterrows():
            with st.container():
                st.markdown(f"""
                    <div class="stcard fade-in">
                        <div class="card-title">{row['Assessment Name']}</div>
                        <div class="tag-container">
                            <span class="tag">{row['Test Type']}</span>
                            <span class="tag">Duration: {row['Duration in mins']} mins</span>
                            <span class="tag">Remote: {row['Remote Testing']}</span>
                            <span class="tag">Adaptive: {row['Adaptive/IRT']}</span>
                        </div>
                        <div class="card-description">{row['Description']}</div>
                        <div class="card-footer">
                            <a href="{row['Relative URL']}" target="_blank" class="card-link">View Details</a>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No matching assessments found. Try different search terms.")
else:
    # Load and display all assessments
    df = load_data()
    if not df.empty:
        # Apply filters
        if test_type_filter:
            df = df[df['Test Type'].isin(test_type_filter)]
        if remote_filter:
            df = df[df['Remote Testing'].str.lower() == 'yes']
        if adaptive_filter:
            df = df[df['Adaptive/IRT'].str.lower() == 'yes']
        if duration_filter:
            df = df[df['Duration in mins'].fillna(0).astype(float) <= duration_filter]
        
        # Sort data
        if sort_by:
            ascending = sort_order == "Ascending"
            df = df.sort_values(by=sort_by, ascending=ascending)
        
        # Prepare and display data
        display_df = prepare_table_data(df)
        
        for _, row in display_df.iterrows():
            with st.container():
                st.markdown(f"""
                    <div class="stcard fade-in">
                        <div class="card-title">{row['Assessment Name']}</div>
                        <div class="tag-container">
                            <span class="tag">{row['Test Type']}</span>
                            <span class="tag">Duration: {row['Duration in mins']} mins</span>
                            <span class="tag">Remote: {row['Remote Testing']}</span>
                            <span class="tag">Adaptive: {row['Adaptive/IRT']}</span>
                        </div>
                        <div class="card-description">{row['Description']}</div>
                        <div class="card-footer">
                            <a href="{row['Relative URL']}" target="_blank" class="card-link">View Details</a>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #333;">
        <p style="color: #666;">Made with ❤️ by SHL Assessment Recommendation System</p>
    </div>
""", unsafe_allow_html=True)