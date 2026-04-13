import pandas as pd

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f'Data loaded successfully from {file_path}')
        return data
    except Exception as e:
        print(f'Error loading data: {e}')
        return None
    
def remove_unused_columns(data):
    """
    Remove unneeded columns, in this case, the id, song name, artist and target columns. 
    """
    columns_to_remove = ['', 'song_title', 'artist', 'target']
    data = data.drop(columns=columns_to_remove, errors='ignore')
    print(f'Columns {columns_to_remove} removed successfully.')
    return data

def remove_first_row(data):
    """
    Remove the first row of the dataset, which contains the column names. 
    """
    data = data.iloc[1:].reset_index(drop=True)
    print('First row removed successfully.')
    return data

def save_processed_data(data, output_path):
    try:
        data.to_csv(output_path, index=False)
        print(f'Processed data saved successfully to {output_path}')
    except Exception as e:
        print(f'Error saving processed data: {e}')

def process_dataset(file_path, save_processed=False, output_path='tmp/processed_data.csv'):
    data = load_data(file_path)
    if data is not None:
        data = remove_unused_columns(data)
        data = remove_first_row(data)
        print('Dataset processed successfully.')
        if save_processed:
            save_processed_data(data, output_path)
        return data
    else:
        print('Failed to process dataset.')
        return None