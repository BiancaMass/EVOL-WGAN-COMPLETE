import os


def find_latest_generator_file(directory):
    # List all files in the directory
    files = os.listdir(directory)

    # Filter out files to only those that are relevant (starts with 'generator-' and ends with '.pt')
    generator_files = [f for f in files if f.startswith('generator-') and f.endswith('.pt')]

    # If no generator files are found, return None
    if not generator_files:
        return None

    # Extract the numbers from the filenames and find the file with the maximum number
    max_file = max(generator_files, key=lambda x: int(x.split('-')[1].split('.pt')[0]))

    # Return the path to the most recent generator file
    return os.path.join(directory, max_file)

