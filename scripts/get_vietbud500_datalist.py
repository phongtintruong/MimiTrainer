import requests
# from dotenv import load_dotenv
import os
from huggingface_hub import login


# load_dotenv()

# hf_token = os.getenv("HF_TOKEN_READ")
# print(hf_token)
login('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')

# Dataset repository on Hugging Face
repo_name = "linhtran92/viet_bud500"

# Hugging Face API URL for the dataset repo
url = f"https://huggingface.co/api/datasets/{repo_name}"

# Send a GET request to fetch the repository information
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    repo_info = response.json()
    print("Repository info fetched successfully.")

    # Lists to store URLs for each split
    train_files = []
    test_files = []
    validation_files = []

    # Process each sibling file from the repo info
    for sibling in repo_info.get('siblings', []):
        rfilename = sibling.get('rfilename', '')
        
        if rfilename.startswith('data/train'):
            # Construct the URL for the train files
            train_files.append(f"https://huggingface.co/datasets/{repo_name}/resolve/main/{rfilename}")
        elif rfilename.startswith('data/test'):
            # Construct the URL for the test files
            test_files.append(f"https://huggingface.co/datasets/{repo_name}/resolve/main/{rfilename}")
        elif rfilename.startswith('data/validation'):
            # Construct the URL for the validation files
            validation_files.append(f"https://huggingface.co/datasets/{repo_name}/resolve/main/{rfilename}")
    
    print('test')
    # Function to save the list of URLs to a file
    def save_to_file(filename, data_list):
        with open(filename, 'w') as file:
            for item in data_list:
                file.write(f"{item}\n")

    # Save the URLs to separate files
    save_to_file('audio/vietbud500_train_files.txt', train_files)
    save_to_file('audio/vietbud500_test_files.txt', test_files)
    save_to_file('audio/vietbud500_validation_files.txt', validation_files)

    print("Files saved successfully.")

else:
    print(f"Failed to fetch repository info. HTTP Status Code: {response.status_code}")