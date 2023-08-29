import requests
from tqdm import tqdm

# Replace with your Dropbox file's public link
file_url = "https://www.dropbox.com/scl/fi/9oxc9n1fqkmz3wrbtjl57/model.onnx?rlkey=gyowu26gsht6qv7emur6r5apv&dl=0"

# Extract the file's direct download link
direct_link = file_url.replace("www.dropbox.com", "dl.dropboxusercontent.com")

# Define the local file path where you want to save the downloaded file
local_file_path = "models/model.onnx"

# Download the file with a progress bar
response = requests.get(direct_link, stream=True)
total_size = int(response.headers.get('content-length', 0))

if response.status_code == 200:
    with open(local_file_path, "wb") as file, tqdm(
        desc=local_file_path,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))
    print("File downloaded successfully.")
else:
    print("Failed to download the file.")