# ASCII Art Banner
print("""

DEV-BY
 ░▒▓██████▓▒░░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░  
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓████████▓▒░▒▓███████▓▒░░▒▓████████▓▒░▒▓███████▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓████████▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░  ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░  ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░░▒▓█▓▒░  ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░ 
Auto-Captioner                                                                                                                                                                                                             
""")

# Link
print("https://linktr.ee/ababiya")
print()  # Add an empty line for better readability

import os

def create_captions():
    # Get all files in the current directory
    files = os.listdir()
    
    # List of common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    
    # Get image files
    image_files = [file for file in files if any(file.lower().endswith(ext) for ext in image_extensions)]
    
    if not image_files:
        print("No image files found in the directory.")
        return
    
    # Ask for input caption once
    caption = input("Enter the caption for all images (separate words with spaces): ")
    
    # Split the caption into words and join with commas
    caption_words = caption.split()
    comma_separated_caption = ", ".join(caption_words)
    
    # Process each image file
    for file in image_files:
        # Get the filename without extension
        name_without_ext = os.path.splitext(file)[0]
        
        # Create a .txt file with the same name as the image
        txt_filename = f"{name_without_ext}.txt"
        
        # Write the comma-separated caption to the .txt file
        with open(txt_filename, 'w') as txt_file:
            txt_file.write(comma_separated_caption)
    
    print(f"Caption files created successfully for {len(image_files)} images.")

# Run the function
create_captions()