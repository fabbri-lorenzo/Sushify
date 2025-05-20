# A small code to change photos extension in .jpg format

import os

def make_jpg(directory, new_extension):
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file has already the new extension
        if filename.endswith(new_extension) == False:
            # Generate the new filename with the new extension
            new_filename = os.path.splitext(filename)[0] + new_extension

            # Rename the file
            os.rename(
                os.path.join(directory, filename),
                os.path.join(directory, new_filename),
            )
            print(f"Renamed '{filename}' to '{new_filename}'")


directory_path = "data/sushi_types/test/gunkan_maki"
new_extension = ".jpg"  # Specify the new extension you want
make_jpg(directory_path, new_extension)
