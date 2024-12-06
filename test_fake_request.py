import requests

# URL for the app
URL = "http://127.0.0.1:5000/upload"

# Simulate file upload
fake_file_content = b"This is a fake ECG file content for testing."
fake_file_path = "fake_ecg.dat"

# Save fake file to disk
with open(fake_file_path, "wb") as f:
    f.write(fake_file_content)

# Prepare the request
with open(fake_file_path, "rb") as file:
    files = {"ecg_file": file}
    response = requests.post(URL, files=files)

# Print the server's response
if response.ok:
    print("Response from server:")
    print(response.json())
else:
    print(f"Error: {response.status_code}")
    print(response.text)
