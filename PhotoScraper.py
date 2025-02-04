import os
import requests

# Importing bird_dict from BirdDictionary.py
from BirdDictionary import bird_dict

# Main directory to save images
MAIN_IMAGE_SAVE_DIRECTORY = "Bird_Images"

# File to track processed birds
PROCESSED_BIRDS_FILE = "processed_birds.txt"

# Google Custom Search API Variables
API_KEY = "AIzaSyDmqrNcRK3pbl8QUc24pQNMrAXakw3J65Y"  # Replace with your API Key
CSE_ID = "f22db98325c5442e8"  # Replace with your Custom Search Engine ID
DAILY_SEARCH_LIMIT = 100  # Limit to the free tier daily queries

# Ensure the main save directory exists
if not os.path.exists(MAIN_IMAGE_SAVE_DIRECTORY):
    os.makedirs(MAIN_IMAGE_SAVE_DIRECTORY)


# Load previously processed bird names
def load_processed_birds():
    if os.path.exists(PROCESSED_BIRDS_FILE):
        with open(PROCESSED_BIRDS_FILE, "r") as file:
            return set(file.read().splitlines())
    return set()


# Save processed bird names
def save_processed_birds(processed_birds):
    with open(PROCESSED_BIRDS_FILE, "w") as file:
        file.write("\n".join(processed_birds))


# Fetch image URLs for a bird using Google Custom Search API
def fetch_image_urls(bird_name, api_key, cse_id, max_results=1):
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": bird_name,  # Query term (bird name)
        "cx": cse_id,  # Custom Search Engine ID
        "key": api_key,  # API key
        "searchType": "image",  # Search for images
        "num": max_results,  # Number of results to retrieve
        "imgType": "photo",  # Filter for photos
    }

    try:
        # Make the request to Google Custom Search API
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        search_results = response.json()

        # Extract image URLs
        image_urls = []
        if "items" in search_results:
            image_urls = [item["link"] for item in search_results["items"]]  # Get direct image URLs

        return image_urls
    except Exception as e:
        print(f"Error fetching image URLs for {bird_name}: {e}")
        return []


# Download images for a specific bird
def download_image(bird_name, image_url):
    # Format the folder name based on the bird name
    folder_name = bird_name.replace(" ", "_")  # Replace spaces with underscores
    subfolder_path = os.path.join(MAIN_IMAGE_SAVE_DIRECTORY, folder_name)

    # Create the subfolder for the bird name
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    # Download the image
    try:
        image_data = requests.get(image_url).content
        # Save the image in the subfolder
        file_path = os.path.join(subfolder_path, f"{folder_name}.jpg")
        with open(file_path, "wb") as image_file:
            image_file.write(image_data)
        print(f"Image for {bird_name} saved at: {file_path}")
    except Exception as e:
        print(f"Failed to download image for {bird_name}: {e}")


# Main function to run the image download process
def process_birds(bird_dict, api_key, cse_id, daily_limit):
    processed_birds = load_processed_birds()
    queries_used_today = 0

    for bird_id, bird_name in bird_dict.items():
        if queries_used_today >= daily_limit:
            print("Daily search limit reached. Exiting for today.")
            break

        if bird_name in processed_birds:
            print(f"Skipping {bird_name}: already processed.")
            continue

        print(f"Searching for photos of: {bird_name}")
        image_urls = fetch_image_urls(bird_name, api_key, cse_id)
        if image_urls:
            # Download the first image URL
            download_image(bird_name, image_urls[0])
            processed_birds.add(bird_name)
            queries_used_today += 1
        else:
            print(f"No image found for {bird_name}.")

    # Save processed birds to a file
    save_processed_birds(processed_birds)


# Run the function
if __name__ == "__main__":
    if not API_KEY or not CSE_ID:
        print("Error: Please provide valid Google API key and Custom Search Engine ID.")
    else:
        process_birds(bird_dict, API_KEY, CSE_ID, DAILY_SEARCH_LIMIT)
