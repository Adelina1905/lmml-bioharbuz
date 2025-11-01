import pandas as pd
import glob
import os

def merge_all_datasets():
    """
    Merge all CSV dataset files into one complete dataset
    """
    # Get all CSV files that match the pattern dataset_part_*.csv
    csv_files = sorted(glob.glob('./BRAZIL/dataset/dataset_part_*.csv'))
    
    print(f"Found {len(csv_files)} CSV files to merge:")
    for file in csv_files:
        print(f"  - {file}")
    
    if not csv_files:
        print("No dataset files found!")
        return
    
    # Read and merge all CSV files
    dataframes = []
    total_rows = 0
    
    for file in csv_files:
        df = pd.read_csv(file)
        print(f"\n{os.path.basename(file)}: {df.shape[0]} rows, {df.shape[1]} columns")
        total_rows += df.shape[0]
        dataframes.append(df)
    
    # Concatenate all dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    print(f"\n{'='*60}")
    print(f"Total rows before merge: {total_rows}")
    print(f"Merged DataFrame shape: {merged_df.shape}")
    
    # Convert timestamp to datetime
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
    
    # Sort by timestamp
    merged_df = merged_df.sort_values(by='timestamp').reset_index(drop=True)
    print(f"Sorted by timestamp")
    
    # Remove duplicates if any
    original_rows = len(merged_df)
    merged_df = merged_df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = original_rows - len(merged_df)
    
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate rows")
    
    print(f"\nFinal DataFrame shape: {merged_df.shape}")
    print(f"Date range: {merged_df['timestamp'].min()} to {merged_df['timestamp'].max()}")
    print(f"Unique series: {merged_df['series_id'].unique()}")
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Statistics by Series:")
    for series_id in sorted(merged_df['series_id'].unique()):
        series_data = merged_df[merged_df['series_id'] == series_id]
        print(f"\n{series_id}:")
        print(f"  Count: {len(series_data)}")
        print(f"  Value range: [{series_data['value'].min():.2f}, {series_data['value'].max():.2f}]")
        print(f"  Mean noise: {series_data['noise_level'].mean():.3f}")
    
    # Save the merged dataset
    output_file = './BRAZIL/dataset/fulldata.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"\n{'='*60}")
    print(f"✓ Merged dataset saved to: {output_file}")
    print(f"✓ Total rows: {len(merged_df)}")
    print(f"✓ Total columns: {len(merged_df.columns)}")
    
    return merged_df

if __name__ == "__main__":
    merge_all_datasets()