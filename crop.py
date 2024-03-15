
from PIL import Image
from glob import glob


def crop_to_square(image_path, folder_name, final_folder):
    # Open the image
    img = Image.open(image_path)
    # Get dimensions of the original image
    width, height = img.size

    # Calculate dimensions for the square crop
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    right = (width + size) // 2
    bottom = (height + size) // 2

    # Crop the image to the square
    cropped_img = img.crop((left, top, right, bottom))

    cropped_img = cropped_img.resize((50, 50))
    # Save or display the cropped image
    cropped_img.save(image_path.replace(folder_name, final_folder))
    # If you want to save the cropped image:


def apply_folder(folder_name, final_folder, function=crop_to_square):
    files = glob(folder_name + "/*")
    print(files)
    for file in files:
        function(file, folder_name, final_folder)


apply_folder("panties2", "crop_panties")