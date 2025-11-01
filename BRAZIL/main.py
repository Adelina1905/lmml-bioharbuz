import pandas as pd
import matplotlib.pyplot as plt

def scatter_plot_data(csv_file, max_noise_level=0.3):
    """
    Create scatter plots for all series with low noise
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter for low noise data points
    df_filtered = df[df['noise_level'] <= max_noise_level]
    
    # Sort by timestamp
    df_filtered = df_filtered.sort_values('timestamp')
    
    print(f"Original data shape: {df.shape}")
    print(f"Filtered data shape (noise <= {max_noise_level}): {df_filtered.shape}")
    print(f"Unique series: {df_filtered['series_id'].unique()}")
    print(f"Date range: {df_filtered['timestamp'].min()} to {df_filtered['timestamp'].max()}")
    
    # Create scatter plot - all series on one plot
    plt.figure(figsize=(15, 8))
    
    # Plot each series with different color
    for series_id in df_filtered['series_id'].unique():
        series_data = df_filtered[df_filtered['series_id'] == series_id]
        plt.scatter(series_data['timestamp'], series_data['value'], 
                   label=series_id, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    
    plt.xlabel('Timestamp', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(f'Time Series Data - All Series (Noise Level ≤ {max_noise_level})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'./BRAZIL/scatter_plot_low_noise_all_series.png', dpi=300, bbox_inches='tight')
    plt.show()
    

# Run the scatter plot for all series with low noise
scatter_plot_data('./BRAZIL/dataset/fulldata.csv', max_noise_level=0.3)