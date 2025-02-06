from PIL import Image
from pathlib import Path


def convert_images_to_jpeg(folder_path, output_folder):
    """
    Converts all images in a folder (and subfolders) to JPEG format.

    :param folder_path: Path to the folder containing images.
    :param output_folder: Path to the output folder to save converted images.
    """
    input_path = Path('') #replace with input folder
    output_path = Path('') #replace with output folder

    # Create the output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Supported input formats (Pillow-supported formats)
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    for file_path in input_path.rglob('*'):  # Recursively find all files
        if file_path.is_file() and file_path.suffix.lower() in supported_formats:
            try:
                # Open and convert the image
                with Image.open(file_path) as img:
                    img = img.convert("RGB")  # Convert to RGB for JPEG compatibility

                    # Generate output path, preserving folder structure
                    relative_path = file_path.relative_to(input_path)
                    jpeg_path = output_path / relative_path.with_suffix('.jpg')
                    jpeg_path.parent.mkdir(parents=True, exist_ok=True)

                    # Save the image as a JPEG
                    img.save(jpeg_path, format="JPEG")
                    print(f"Converted: {file_path} -> {jpeg_path}")
            except (IOError, OSError) as e:
                print(f"Skipping {file_path}: {e}")
        else:
            print(f"Skipping unsupported file: {file_path}")


# Example usage
folder_path = r'C:\path\to\input\images'  # Replace with your folder path
output_folder = r'C:\path\to\output\images'  # Replace with your output folder path
convert_images_to_jpeg(folder_path, output_folder)
