import gdown

file_id = "1Fdvy1_TJwvHFsximQ6nHn0Np4q6J7VUw"
output_name = "clip-data.zip"

url = f"https://drive.google.com/uc?id={file_id}"

gdown.download(url, output_name, quiet=False)