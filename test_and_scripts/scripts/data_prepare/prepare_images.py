from pathlib import Path
import os
from PIL import Image

def convert_image_to_jpeg(image_file: Path):
    try:
        with Image.open(image_file) as img:
            if img.mode == 'P':
                img = img.convert('RGBA')

            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            else:
                img = img.convert('RGB')

            img.save(image_file.with_suffix('.jpg'), format='JPEG')
            if image_file.suffix.lower() != '.jpg':
                os.remove(image_file)
    except Exception as e:
        print(f"Cannot open {image_file}: {e}")

def main():
    images_folder = Path('images')
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

    for image_file in images_folder.rglob('*'):
        if image_file.suffix.lower() in image_extensions:
            convert_image_to_jpeg(image_file)

if __name__ == "__main__":
    main()
