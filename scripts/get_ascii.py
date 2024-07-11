import numpy as np
from PIL import Image
import os
import argparse
import json
args = None

def remove_white_background(image, threshold=240):
    # Convert image to RGBA if it isn't already
    image = image.convert("RGBA")
    data = np.array(image)
    
    # Create a mask for non-white pixels
    r, g, b, a = data.T
    white_areas = (r > threshold) & (g > threshold) & (b > threshold)
    
    # Set alpha channel to 0 for white pixels
    data[..., -1] = np.where(white_areas.T, 0, 255)
    
    return Image.fromarray(data)
def remove_space_only_lines(ascii_art):
    first_line = ascii_art[0] if ascii_art[0].isspace() else None
    last_line = ascii_art[-1] if ascii_art[-1].isspace() else None
    middle_lines = [line for line in ascii_art if not line.isspace()]

    if first_line:
        middle_lines.insert(0, first_line)
    if last_line:
        middle_lines.append(last_line)

    return middle_lines


def image_to_ascii(image, width=100, ascii_chars=' .:-=+*#%@'):
    aspect_ratio = image.height / image.width
    height = int(aspect_ratio * width * 0.55)  # Adjust for character aspect ratio
    image = image.resize((width, height))
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Convert pixels to ASCII
    pixels = np.array(image)
    bins = np.linspace(0, 255, len(ascii_chars) + 1)
    indices = np.digitize(pixels, bins) - 1
    indices = np.clip(indices, 0, len(ascii_chars) - 1)
    ascii_img = np.array(list(ascii_chars))[indices]
    ascii_lines = [''.join(row) for row in ascii_img]
    ascii_lines = remove_space_only_lines(ascii_lines)
    
    for index, line in enumerate(ascii_lines):
        ascii_lines[index] = line[11:85]

    return '\n'.join(ascii_lines)

def png_to_ascii(file_path, output_width=100, ascii_chars=' .:-=+*#%@', bg_threshold=240):
    # Open the image
    image = Image.open(file_path)
    
    ascii_art = image_to_ascii(image, output_width, ascii_chars)
    
    return ascii_art


def parse_dir():
    ascii_array = [] 

    with open(args.ascii, 'w') as f:
        f.write('')

    for _, _, files in os.walk(args.dir):
        files.sort()

        for file in files: 
            read_path = os.path.join(args.dir, file)
            write_path = os.path.join(args.write_bg_dir, file)

            img = Image.open(read_path)
            img = remove_white_background(image=img)
            
            img.save(write_path)

            ascii = image_to_ascii(img)
            ascii_array.append(ascii)
            with open(args.ascii, 'a') as f:
                f.write('\n')
                f.write(ascii)

    with open('ascii_art.json', 'w') as f:
        json.dump(ascii_array, f)

    # Optionally, also save as a JavaScript file
    js_content = f"const asciiArt = {json.dumps(ascii_array, indent=2)};"
    
    with open('ascii_art.js', 'w') as f:
        f.write(js_content)

def main():
    global args
    parser = argparse.ArgumentParser(description="Convert image to ascii")
    parser.add_argument('--dir', type=str, default='./data/banana', help="Directory path. Default is './data/banana'")
    parser.add_argument('--write_bg_dir', type=str, default='./data/banana_clean', help="Directory path. Default is './data/banana'")
    parser.add_argument('--ascii', type=str, default='./data/ascii.txt', help="Directory path. Default is './data/banana'")

    args = parser.parse_args()
    parse_dir()

if __name__ == '__main__':
    main()


