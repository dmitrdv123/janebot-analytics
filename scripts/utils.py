import os
import pandas as pd

# Create a function to ensure directories exist
def ensure_directory(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)

def load_data(base_folder_path: str, file_extension: str = '.csv'):
  '''
  Downloads all files from subfolders (organized by date) inside a given folder and combines them into a single dataset (DataFrame).

  Args:
  - base_folder_path (str): Path to the base folder containing subfolders (named by date).
  - file_extension (str): Extension of files to download (default is '.csv').

  Returns:
  - pd.DataFrame: Combined dataset of all files in the subfolders.
  '''
  all_files = []

  # Check if base folder exists
  if not os.path.exists(base_folder_path):
    raise FileNotFoundError(f'The base folder {base_folder_path} does not exist.')

  # Iterate over the subfolders (which represent dates)
  for subfolder_name in os.listdir(base_folder_path):
    subfolder_item = os.path.join(base_folder_path, subfolder_name)

    # Check if the subfolder is indeed a directory (i.e., not a file)
    if os.path.isdir(subfolder_item):
      # Iterate over the files in the subfolder
      for filename in os.listdir(subfolder_item):
        # Filter based on file extension (e.g., CSV)
        if filename.endswith(file_extension):
          file_path = os.path.join(subfolder_item, filename)
          # Read file (assuming CSV for simplicity)
          df = pd.read_csv(file_path)
          all_files.append(df)
    elif os.path.isfile(subfolder_item):
      if subfolder_item.endswith(file_extension):
        df = pd.read_csv(subfolder_item)
        all_files.append(df)

  # Combine all files into a single DataFrame
  if all_files:
    combined_data = pd.concat(all_files, ignore_index=True)
    return combined_data
  else:
    raise ValueError('No files found in the subfolders with the specified extension.')
