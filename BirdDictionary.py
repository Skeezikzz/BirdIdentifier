import requests
from bs4 import BeautifulSoup
import re

def read_website_content(url):
    """
    Fetches and prints the entire content of a website.

    Args:
        url (str): The website URL to fetch.

    Returns:
        str: The content of the website if successful, or an error message otherwise.
    """
    try:
        # Fetch content from the website
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.text  # Return the website content as plain text

    except requests.exceptions.RequestException as e:
        return f"Error fetching the website: {e}"


## Example usage
website_url = "http://www.ofo.ca/site/page/view/checklist.checklist"  # Replace with your desired URL

# Read and print the content of the website
website_content = read_website_content(website_url)
content_lines = website_content.splitlines()
#print(website_content)

bird_dict = {}

bird_counter = 1

for line in content_lines:
    line = line.strip()
    bird_names = re.findall(r'<p>(.*?)\s+\(', line)
    bird_names = [name.replace("*", "") for name in bird_names]
    bird_names = [name.replace("L/C/S", "") for name in bird_names]
    bird_names = [name.replace("L/(S)", "") for name in bird_names]
    bird_names = [name.replace("L/(C)/S", "") for name in bird_names]
    bird_names = [name.replace("C/S", "") for name in bird_names]
    bird_names = [name.replace("L/(C)/(S)", "") for name in bird_names]
    bird_names = [name[:-1].strip() if name.endswith("S") else name for name in bird_names]
    if bird_names != []:
        for bird in bird_names:
            bird_name_cleaned = bird.strip()
            bird_dict [f"Bird{bird_counter}"] = bird_name_cleaned
            bird_counter += 1


bird_dict = {key: bird_dict[key] for key in list(bird_dict.keys())[3:]}

print(bird_dict)