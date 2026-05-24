import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_loss_ratio_by_province(df: pd.DataFrame, save_path: str = None):
    """Loss Ratio by Province"""
    if 'Province' not in df.columns or 'LossRatio' not in df.columns:
        print("Required columns not found.")
        return
    
    plt.figure(figsize=(12, 6))
    province_lr = df.groupby('Province')['LossRatio'].mean().sort_values(ascending=False)
    sns.barplot(x=province_lr.index, y=province_lr.values, palette='viridis')
    plt.title('Average Loss Ratio by Province', fontsize=14)
    plt.xlabel('Province')
    plt.ylabel('Loss Ratio')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_premium_vs_claims(df: pd.DataFrame, hue='Province'):
    """Scatter plot of Premium vs Claims"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='TotalPremium', y='TotalClaims', hue=hue, alpha=0.6)
    plt.title('Total Premium vs Total Claims')
    plt.xlabel('Total Premium')
    plt.ylabel('Total Claims')
    plt.tight_layout()
    plt.show()

def top_vehicle_makes(df: pd.DataFrame, n=10):
    """Top vehicle makes by claims"""
    if 'Make' in df.columns:
        makes = df.groupby('Make')['TotalClaims'].sum().sort_values(ascending=False).head(n)
        print("\nTop Vehicle Makes by Total Claims:")
        print(makes)