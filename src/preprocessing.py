import pandas as pd
import numpy as np

def handle_missing_values(df, strategy='interpolate'):
    """
    Handle missing values in the dataset
    
    Args:
        df: pandas DataFrame
        strategy: 'interpolate', 'forward_fill', 'median', or 'drop'
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if strategy == 'interpolate':
        # Linear interpolation for time-series data (best for year-over-year health data)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'Year':
                df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
    
    elif strategy == 'forward_fill':
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
    
    elif strategy == 'median':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    elif strategy == 'drop':
        df_clean = df_clean.dropna()
    
    return df_clean

def detect_outliers(df, column, method='iqr', threshold=1.5):
    """
    Detect outliers in a column
    
    Args:
        df: pandas DataFrame
        column: Column name
        method: 'iqr' or 'zscore'
        threshold: IQR multiplier or z-score threshold
        
    Returns:
        Boolean mask of outliers
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        outliers = z_scores > threshold
    
    return outliers

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize numerical columns
    
    Args:
        df: pandas DataFrame
        columns: List of columns to normalize (None = all numeric)
        method: 'minmax' or 'standard'
        
    Returns:
        DataFrame with normalized columns
    """
    df_norm = df.copy()
    
    if columns is None:
        columns = df_norm.select_dtypes(include=[np.number]).columns
        columns = [col for col in columns if col != 'Year']
    
    if method == 'minmax':
        for col in columns:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    
    elif method == 'standard':
        for col in columns:
            mean = df_norm[col].mean()
            std = df_norm[col].std()
            df_norm[col] = (df_norm[col] - mean) / std
    
    return df_norm

def create_year_categories(df):
    """
    Create categorical year periods for analysis
    
    Args:
        df: pandas DataFrame with 'Year' column
        
    Returns:
        DataFrame with 'Period' column added
    """
    df_cat = df.copy()
    
    conditions = [
        (df_cat['Year'] >= 2000) & (df_cat['Year'] < 2010),
        (df_cat['Year'] >= 2010) & (df_cat['Year'] < 2020),
        (df_cat['Year'] >= 2020)
    ]
    
    choices = ['2000-2009', '2010-2019', '2020+']
    
    df_cat['Period'] = np.select(conditions, choices, default='Unknown')
    
    return df_cat

# Test
if __name__ == "__main__":
    from data_loader import load_maternal_health_data
    
    df = load_maternal_health_data()
    
    print("Original missing values:")
    print(df.isnull().sum())
    
    df_clean = handle_missing_values(df, strategy='interpolate')
    
    print("\nAfter interpolation:")
    print(df_clean.isnull().sum())
    
    print("\nSample cleaned data:")
    print(df_clean.head(10))
    
    # Save processed data
    df_clean.to_csv('data/processed/maternal_health_cleaned.csv', index=False)
    print("\n✅ Saved to data/processed/maternal_health_cleaned.csv")