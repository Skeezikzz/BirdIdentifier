import os
import json
import time
import random
import requests
from PIL import Image
from io import BytesIO
from BirdDictionary import bird_dict

# Constants
PROCESSED_BIRDS_FILE = 'processed_birds.txt'  # File to save processed bird URLs
IMAGES_PER_BIRD = 10  # Number of images to download per script run
MAX_IMAGES_PER_BIRD = 100  # Maximum number of total images for each bird


def fetch_image_urls(bird_name, api_key, cse_id, max_results=10, start_index=1, query=None, site=None):
    """Fetch image URLs from Google's Custom Search API"""
    search_url = "https://www.googleapis.com/customsearch/v1"

    # Prepare query
    if query is None:
        query = f"{bird_name} bird photo"
    if site:
        query += f" site:{site}"

    # Request parameters
    params = {
        "q": query,
        "cx": cse_id,
        "key": api_key,
        "searchType": "image",
        "num": max_results,
        "imgType": "photo",
        "start": start_index  # Pagination
    }

    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        search_results = response.json()
        return [item["link"] for item in search_results.get("items", [])]
    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:  # Too many requests
            print("HTTP 429 Too Many Requests. Retrying after delay...")
            time.sleep(30)  # Throttle when hitting rate limits
        else:
            print(f"HTTP Error: {e}")
    except Exception as e:
        print(f"Error fetching image URLs: {e}")

    return []  # Return empty list if there's an error


def download_image(bird_name, url, index):
    """
    Download an image from the given URL and save it locally.

    :param bird_name: The name of the bird (used for folder organization)
    :param url: The image URL to download
    :param index: Image index for naming the file
    """
    try:
        # Make the HTTP request to fetch the image
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Save the image to the appropriate directory
        folder_name = f"bird_images/{bird_name.replace(' ', '_')}"
        os.makedirs(folder_name, exist_ok=True)  # Create the directory if it doesn't exist
        file_path = os.path.join(folder_name, f"image_{index}.jpg")  # Save as a .jpg file

        # Write the image file
        with open(file_path, 'wb') as f:
            f.write(response.content)

        print(f"Saved {bird_name} image {index} to: {file_path}")

        # Optional: Verify if the downloaded image is valid
        try:
            Image.open(BytesIO(response.content)).verify()  # Check for corrupted files
            print(f"Image {index} for {bird_name} verified successfully.")
        except Exception as verify_error:
            print(f"Image verification failed for {file_path}: {verify_error}")
            os.remove(file_path)  # Remove invalid files if any

    except requests.exceptions.RequestException as e:
        print(f"Failed to download image {index} for {bird_name} from {url}: {e}")
    except Exception as e:
        print(f"Unexpected error occurred while downloading {url}: {e}")


def load_processed_birds():
    """Load previously processed birds and their URLs from a file"""
    try:
        with open(PROCESSED_BIRDS_FILE, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}  # Return empty dictionary if no file exists
    except Exception as e:
        print(f"Error loading processed birds file: {e}")
        return {}


def save_processed_birds(processed_birds):
    """Save processed birds and their URLs to a file"""
    try:
        with open(PROCESSED_BIRDS_FILE, 'w') as file:
            json.dump(processed_birds, file, indent=4)
    except Exception as e:
        print(f"Error saving processed birds data: {e}")


def process_birds(bird_dict, api_key, cse_id, max_images_per_bird, reset_processed=False):
    """Process and download images for all birds"""
    if reset_processed:
        processed_birds = {}
    else:
        processed_birds = load_processed_birds()

    queries_used_today = 0  # Track total queries used for the current run

    search_sites = ["flickr.com", "pexels.com", "unsplash.com"]  # Example additional sources
    query_variations = [
        "wild photo", "flying", "close up", "colorful", "rare bird"
    ]

    print("\n--- Starting Image Fetching ---")
    for bird_key, bird_name in bird_dict.items():
        # Skip if the bird already has the maximum allowed images
        if len(processed_birds.get(bird_name, [])) >= max_images_per_bird:
            print(f"Skipping {bird_name}: Already has {len(processed_birds[bird_name])} images.")
            continue

        processed_birds.setdefault(bird_name, [])  # Ensure the bird has an entry
        previously_processed_urls = set(processed_birds[bird_name])  # To filter duplicates

        # Keep track of unique images processed for this bird
        total_downloaded = len(processed_birds[bird_name])
        print(f"Processing {bird_name}. Current total: {total_downloaded} images.")

        for start in range(1, 91, 10):  # Pagination (start = 1, 11, 21, ..., max 90 as per API limit)
            if total_downloaded >= max_images_per_bird:
                break  # Stop once we reach the max per bird

            # Randomize query to fetch unique results
            query_suffix = random.choice(query_variations)
            random_site = random.choice(search_sites)
            query = f"{bird_name} {query_suffix} photo"
            print(f"Fetching images with query: '{query}' and site: {random_site}")

            # Fetch image URLs for this bird
            image_urls = fetch_image_urls(
                bird_name, api_key, cse_id, max_results=10, start_index=start, query=query, site=random_site
            )
            if not image_urls:
                print(f"No new results for {bird_name} on page {start}.")
                break  # Exit early if no results

            # Filter out previously processed URLs
            new_image_urls = [url for url in image_urls if url not in previously_processed_urls]
            print(f"Fetched URLs: {len(new_image_urls)} new unique URLs on page {start}.")

            for idx, url in enumerate(new_image_urls, start=total_downloaded + 1):
                if total_downloaded >= max_images_per_bird:
                    break
                try:
                    # Download and mark the image as processed
                    download_image(bird_name, url, idx)
                    processed_birds[bird_name].append(url)
                    total_downloaded += 1
                except Exception as e:
                    print(f"Failed to process {url}: {e}")

            # Save progress periodically
            save_processed_birds(processed_birds)

            # Respect rate limits (10 QPS -> delay ~0.1 seconds per call)
            print("Throttling: Waiting 1 second between requests.")
            time.sleep(1)  # Short delay to prevent rate limit issues

    print("--- Finished Processing Birds ---")
    print(f"Total queries used today: {queries_used_today}")
    return


# Main entry (example usage, add your API keys and bird data)
if __name__ == "__main__":
    API_KEY = "AIzaSyDmqrNcRK3pbl8QUc24pQNMrAXakw3J65Y"  # Replace with your Google Custom Search API key
    CSE_ID = "f22db98325c5442e8"  # Replace with your Custom Search Engine ID



    process_birds(bird_dict, API_KEY, CSE_ID, MAX_IMAGES_PER_BIRD)
