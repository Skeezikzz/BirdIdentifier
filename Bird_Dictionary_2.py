import os
import re

bird_scientific_names = []

## Directory path
directory = "" #replace with your directory

# Check if directory exists
if not os.path.exists(directory):
    print(f"Directory '{directory}' does not exist. Please check the path.")
    exit(1)

# Walk through the directory
for dirname, subdirs, filenames in os.walk(directory):
    print(f"Currently traversing: {dirname}")  # Debug: What directory is being processed
    print(f"Subdirectories: {subdirs}")  # Debug: List of subdirectories found

    # Process each subdirectory
    for subdir in subdirs:
        print(f"Processing subdirectory: {subdir}")  # Debug: Current subdirectory name

        # Extract scientific names based on 'bird_' prefix
        if "bird_" in subdir:
            scientific_name = subdir.split("bird_")[1]
            bird_scientific_names.append(scientific_name)

        # Regex fallback: Match "Capitalized_lowercase"
        else:
            match = re.search(r'[A-Z][a-z]+_[a-z]+', subdir)
            if match:
                scientific_name = match.group(0)
                bird_scientific_names.append(scientific_name)

# Remove duplicates
bird_scientific_names = list(dict.fromkeys(bird_scientific_names))

# Print final results
print("Bird Scientific Names:", bird_scientific_names)
