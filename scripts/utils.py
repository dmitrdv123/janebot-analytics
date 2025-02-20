import os

# Create a function to ensure directories exist
def ensure_directory(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)
