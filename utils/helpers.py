import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def assess_data_quality(df, name="Dataset"):
    print(f"\n{'='*60}")
    print(f"DATA QUALITY ASSESSMENT: {name}")
    print(f"{'='*60}\n")
    
    print(f"📊 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    
    print("🔍 Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df.to_string())
    else:
        print("✅ No missing values found")
    
    print(f"\n📋 Data Types:")
    print(df.dtypes.value_counts())
    
    print(f"\n🔢 Numeric Columns: {df.select_dtypes(include=[np.number]).columns.tolist()}")
    print(f"📝 Categorical Columns: {df.select_dtypes(include=['object', 'category']).columns.tolist()}")
    
    print(f"\n🔄 Duplicate Rows: {df.duplicated().sum():,}")
    
    print(f"\n{'='*60}\n")
    
    return missing_df

def plot_missing_data(df, title="Missing Data Visualization"):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    
    if len(missing) == 0:
        print("✅ No missing data to visualize")
        return
    
    plt.figure(figsize=(10, 6))
    missing.plot(kind='barh', color='coral')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Number of Missing Values')
    plt.ylabel('Columns')
    plt.tight_layout()
    plt.show()

def create_time_features(df, date_column):
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['day_name'] = df[date_column].dt.day_name()
    df['week_of_year'] = df[date_column].dt.isocalendar().week
    df['quarter'] = df[date_column].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    return df

def plot_sales_trends(df, date_col, sales_col, title="Sales Trends Over Time"):
    plt.figure(figsize=(14, 6))
    plt.plot(df[date_col], df[sales_col], linewidth=2, color='steelblue')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sales', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def compare_models(results_dict, metric='MAE'):
    model_names = list(results_dict.keys())
    scores = [results_dict[model][metric] for model in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, scores, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'][:len(model_names)])
    plt.title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold')
    plt.ylabel(metric, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def print_model_metrics(y_true, y_pred, model_name="Model"):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n{'='*60}")
    print(f"📊 {model_name} Performance Metrics")
    print(f"{'='*60}")
    print(f"Mean Absolute Error (MAE):    {mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
    print(f"R² Score:                     {r2:.4f}")
    print(f"Mean Absolute % Error (MAPE):  {mape:.2f}%")
    print(f"{'='*60}\n")
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

def save_figure(fig, filename, output_dir='../outputs/visualizations'):
    import os
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filepath}")

def set_plot_style():
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
