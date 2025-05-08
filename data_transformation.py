import pandas as pd
import re

def extract_duration(length_str):
    """Extract duration in minutes from Assessment Length string."""
    if pd.isna(length_str):
        return None
    match = re.search(r'(\d+)', str(length_str))
    return int(match.group(1)) if match else None

def extract_skills_from_test_type(test_type):
    """Extract potential skills from test type."""
    if pd.isna(test_type):
        return ""
    
    # Map test types to relevant skills
    skill_mapping = {
        'Knowledge & Skills': 'Technical Skills',
        'Simulations': 'Practical Skills',
        'Cognitive': 'Problem Solving',
        'Personality': 'Soft Skills'
    }
    
    return skill_mapping.get(test_type, "")

def transform_data():
    # Read the data
    df = pd.read_csv('data1.csv')
    
    # Extract duration from Assessment Length
    df['Duration in mins'] = df['Assessment Length'].apply(extract_duration)
    
    # Extract skills from Test Type
    df['Skills'] = df['Test Type'].apply(extract_skills_from_test_type)
    
    # Generate Description
    df['Description'] = df.apply(
        lambda row: (
            f"The '{row['Assessment Name']}' is a {row['Test Type'] if pd.notna(row['Test Type']) else 'General'} assessment. "
            f"Duration is {row['Duration in mins']} mins if specified. "
            f"It supports remote testing: {row['Remote Testing']} and adaptive format: {row['Adaptive/IRT']}. "
            f"Primary skills assessed: {row['Skills']}."
        ).strip(),
        axis=1
    )
    
    # Select and rename columns
    df = df[[
        'Assessment Name',
        'Relative URL',
        'Duration in mins',
        'Remote Testing',
        'Adaptive/IRT',
        'Test Type',
        'Skills',
        'Description'
    ]]
    
    # Save transformed data
    df.to_csv('transformed_data1.csv', index=False)
    print("Successfully transformed data and saved to transformed_data1.csv")
    
    # Try to combine with SHL_catalog.csv if it exists and is not empty
    try:
        shl_df = pd.read_csv('SHL_catalog.csv')
        if not shl_df.empty:
            combined_df = pd.concat([df, shl_df], ignore_index=True)
            combined_df.to_csv('combined_assessments.csv', index=False)
            print("Successfully combined with SHL_catalog.csv")
        else:
            print("SHL_catalog.csv is empty, skipping combination step")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("SHL_catalog.csv not found or empty, skipping combination step")

if __name__ == "__main__":
    transform_data() 