import pandas as pd
import os

def load_maternal_health_data(filepath=None):
    """
    Load Bhutan maternal health dataset
    
    Args:
        filepath: Path to CSV file (optional)
        
    Returns:
        pandas DataFrame
    """
    if filepath is None:
        # Try multiple possible paths
        possible_paths = [
            "data/raw/cleaned_maternal_health_data.csv",  # From project root
            "../data/raw/cleaned_maternal_health_data.csv",  # From notebooks/
            "../../data/raw/cleaned_maternal_health_data.csv"  # From deeper folders
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                filepath = path
                break
        
        if filepath is None:
            raise FileNotFoundError(
                "Data file not found. Tried:\n" + 
                "\n".join(f"  - {p}" for p in possible_paths)
            )
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns from {filepath}")
    
    return df

def validate_data(df):
    """
    Validate the dataset structure and check for issues
    
    Args:
        df: pandas DataFrame
        
    Returns:
        Dictionary with validation results
    """
    validation_report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
        "data_types": df.dtypes.to_dict(),
        "year_range": (df['Year'].min(), df['Year'].max()) if 'Year' in df.columns else None
    }
    
    return validation_report

def get_column_summary(df):
    """
    Get summary statistics for all columns
    
    Args:
        df: pandas DataFrame
        
    Returns:
        DataFrame with summary statistics
    """
    summary = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.values,
        'Non-Null Count': df.count().values,
        'Null Count': df.isnull().sum().values,
        'Null %': (df.isnull().sum() / len(df) * 100).round(2).values
    })
    
    return summary

# Quick test
if __name__ == "__main__":
    df = load_maternal_health_data()
    print("\nDataset Preview:")
    print(df.head())
    
    print("\nValidation Report:")
    report = validate_data(df)
    for key, value in report.items():
        print(f"{key}: {value}")
    
    print("\nColumn Summary:")
    print(get_column_summary(df))