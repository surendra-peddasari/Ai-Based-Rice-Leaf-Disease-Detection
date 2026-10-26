import os
from PIL import Image

# Path directly to the 'Data' folder containing the 3 subfolders
DATA_DIR = os.path.join("rice_leaf_dataset", "Data")

class_counts = {}
corrupted_files = []

# Loop through each of the 3 disease folders
for class_name in os.listdir(DATA_DIR):
    class_folder = os.path.join(DATA_DIR, class_name)
    
    if os.path.isdir(class_folder):
        # Collect image files
        images = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_counts[class_name] = len(images)
        
        # Verify each image is readable and not corrupted
        for img_name in images:
            img_path = os.path.join(class_folder, img_name)
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception:
                corrupted_files.append(img_path)

print("\n--- Image Dataset Breakdown ---")
for cls, count in class_counts.items():
    print(f"Class '{cls}': {count} images")

print(f"\nTotal Classes: {len(class_counts)}")
print(f"Corrupted Images Found: {len(corrupted_files)}")