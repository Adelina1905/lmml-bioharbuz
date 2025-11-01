# Time Series Data Processing

## Overview
This project processes and visualizes time series data from multiple CSV files containing sensor readings with noise levels.

## Files

- **`merge_data.py`** - Merges all `dataset_part_*.csv` files into a single `fulldata.csv`
- **`main.py`** - Creates scatter plots of the time series data with noise filtering
- **`fulldata.csv`** - Complete merged dataset (generated)

## Data Structure

Each CSV file contains:
- `timestamp` - Date and time of the reading
- `series_id` - Identifier for the data series (series_1, series_2, etc.)
- `value` - Measured value
- `noise_level` - Noise level (0.0 to 1.0)

## Usage

### 1. Merged all datasets
```bash
python merge_data.py
```
This creates `fulldata.csv` containing all data sorted by timestamp.

### 2. Visualize the data
```bash
python main.py
```
This generates scatter plots showing only low-noise data points (noise_level ≤ 0.3) which results in showing the secret letters.

## Customization

In `main.py`, adjust the `max_noise_level` parameter to change the noise threshold:
```python
scatter_plot_data('./BRAZIL/dataset/fulldata.csv', max_noise_level=0.3)
```
- `0.0` = No noise only
- `0.5` = Medium noise allowed
- `1.0` = All data points

## Output

- `fulldata.csv` - Merged and sorted complete dataset
- `scatter_plot_low_noise_all_series.png` - Visualization of filtered data