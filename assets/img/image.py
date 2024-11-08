# Fill all transparent pixels with white color in this directory
from PIL import Image
import os

def transparency_eliminate(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            image = Image.open(os.path.join(directory, filename))
            if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
                background = Image.new(image.mode[:-1], image.size, 'white')
                background.paste(image, image.split()[-1])
                background.save(os.path.join(directory, filename))
                print(f"Transparency removed from {filename}")

# Example usage
transparency_eliminate('assets/img')