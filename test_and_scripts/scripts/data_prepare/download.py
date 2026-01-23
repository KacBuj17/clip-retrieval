import gdown

file_id = "1gUpl_zYBJXl8GA4VMVWIYUpjOuoYWcJh"
output_name = "clip-data.zip"

url = f"https://drive.google.com/uc?id={file_id}"

gdown.download(url, output_name, quiet=False)