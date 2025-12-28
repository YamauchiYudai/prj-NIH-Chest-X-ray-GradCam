import os

def find_image_path(filename, data_root="./data"):
    """
    Search for an image file in the NIH Chest X-ray dataset directory structure.
    
    The expected structure is:
    data/images_001/images/{filename}
    data/images_002/images/{filename}
    ...
    data/images_012/images/{filename}
    
    Args:
        filename (str): The name of the image file (e.g., '00000001_000.png').
        data_root (str): The root directory where image folders are located.
        
    Returns:
        str: The full path to the image if found, else None.
    """
    # Check images_001 to images_012
    for i in range(1, 13):
        sub_dir = f"images_{i:03d}"
        path = os.path.join(data_root, sub_dir, "images", filename)
        if os.path.exists(path):
            return path
            
    return None
