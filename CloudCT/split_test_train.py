import os
import random
import shutil
from pathlib import Path

# 1. Set your paths
source_dir = "/wdata/tamarsd/NN_Data/circ_configuration/all_dataset/" # Folder where your 6146 .pkl files are right now
output_dir = "/wdata/tamarsd/NN_Data/circ_configuration_splits/"   # Where you want the new folders created

# 2. Define the exact split numbers (80 / 10 / 10)
num_train = 4916
num_val = 615
# The rest will go to test (615)

# 3. Create the output directories if they don't exist
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

# 4. Grab all .pkl files from the source directory
all_files = [f for f in os.listdir(source_dir) if f.endswith('.pkl')]

# Quick safety check
print(f"Found {len(all_files)} .pkl files.")

# 5. Shuffle the files randomly so your splits are unbiased
random.seed(42) # Set a seed for reproducibility 
random.shuffle(all_files)

# 6. Slice the shuffled list into your three sets
train_files = all_files[:num_train]
val_files = all_files[num_train : num_train + num_val]
test_files = all_files[num_train + num_val:]

# 7. Helper function to copy files
def copy_files(file_list, destination_folder):
    for file_name in file_list:
        src_path = os.path.join(source_dir, file_name)
        dst_path = os.path.join(output_dir, destination_folder, file_name)
        shutil.copy2(src_path, dst_path) # copy2 preserves metadata

# 8. Execute the copy process
print("Copying files to train...")
copy_files(train_files, 'train')

print("Copying files to val...")
copy_files(val_files, 'val')

print("Copying files to test...")
copy_files(test_files, 'test')

print("Done! Your .pkl files are now split.")