import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def save_processed_data(df, filename):
    """Save cleaned data to processed folder"""
    os.makedirs('data/processed', exist_ok=True)
    filepath = f"data/processed/{filename}"
    df.to_csv(filepath, index=False)
    print(f"✅ Saved to {filepath}")
    return filepath

def plot_trend_over_years(df, column, title=None, save_path=None):
    """
    Plot trend of a health indicator over years
    
    Args:
        df: DataFrame with 'Year' column
        column: Column to plot
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(14, 6))
    
    plt.plot(df['Year'], df[column], marker='o', linewidth=2, markersize=6)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel(column, fontsize=12)
    plt.title(title or f'{column} Trend (2000-2023)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_multiple_trends(df, columns, title="Health Indicators Over Time"):
    """
    Plot multiple indicators on the same chart
    
    Args:
        df: DataFrame with 'Year' column
        columns: List of columns to plot
        title: Plot title
    """
    plt.figure(figsize=(14, 8))
    
    for col in columns:
        if col in df.columns and col != 'Year':
            plt.plot(df['Year'], df[col], marker='o', label=col, linewidth=2)
    
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def correlation_heatmap(df, save_path=None):
    """
    Generate correlation heatmap for numeric columns
    
    Args:
        df: pandas DataFrame
        save_path: Path to save figure
    """
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix - Maternal Health Indicators', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def calculate_growth_rate(df, column):
    """
    Calculate year-over-year growth rate
    
    Args:
        df: DataFrame with 'Year' and target column
        column: Column to calculate growth for
        
    Returns:
        DataFrame with growth rate column
    """
    df_growth = df.copy()
    df_growth[f'{column}_growth_%'] = df_growth[column].pct_change() * 100
    return df_growth

def summary_statistics_report(df):
    """
    Generate comprehensive summary statistics
    
    Args:
        df: pandas DataFrame
        
    Returns:
        DataFrame with detailed statistics
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    stats = pd.DataFrame({
        'Mean': df[numeric_cols].mean(),
        'Median': df[numeric_cols].median(),
        'Std Dev': df[numeric_cols].std(),
        'Min': df[numeric_cols].min(),
        'Max': df[numeric_cols].max(),
        'Range': df[numeric_cols].max() - df[numeric_cols].min()
    })
    
    return stats.round(2)

# Test
if __name__ == "__main__":
    from data_loader import load_maternal_health_data
    from preprocessing import handle_missing_values
    
    df = load_maternal_health_data()
    df_clean = handle_missing_values(df, strategy='interpolate')
    
    print("Summary Statistics:")
    print(summary_statistics_report(df_clean))
    
    # Plot a trend
    plot_trend_over_years(df_clean, 'Births attended by skilled health personnel (%)')
    
    # Correlation heatmap
    correlation_heatmap(df_clean)