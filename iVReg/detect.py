import os
import pydicom
import numpy as np
from skimage.transform import resize
import pandas as pd


# Load the CSV file into a pandas DataFrame
df = pd.read_csv(r"D:\Project_vk\results.csv")

# Function to extract row values based on the input class
def extract_row_values(input_class):
    # Filter rows based on the input class
    selected_rows = df[df['class'] == input_class]
    
    # Extract xmin, ymin, xmax, ymax values from the selected row
    if not selected_rows.empty:
        row_values = selected_rows.iloc[0][['xmin', 'ymin', 'xmax', 'ymax']].values
        return tuple(row_values)
    else:
        return None

row_values = extract_row_values(input_class)
row_values = tuple(int(value) for value in row_values)

if row_values:
    print("Extracted row values:", row_values)
else:
    print("No matching rows found for the input class.")



#save cropped

import os
import pydicom
import numpy as np
from skimage.transform import resize

# Define the directory containing DICOM files for extraction
dicom_dir = r"D:\Project_vk\Registration\Sorted_DEC2"

# Define the directory to save the extracted, cropped, and resized DICOM files
output_dir = r"D:\Project_vk\Registration\CROP_DEC2"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the range of image numbers to extract and crop
start_index = row_values[1]
end_index = row_values[3]
bbox_y_start = row_values[0]
bbox_y_end = row_values[2]
bbox_x_start = 215
bbox_x_end = 303

# Define the target size for resizing
target_size = (60, 60)

# Extract, crop, resize, and save DICOM files
for i in range(start_index, end_index + 1):
    # Define input and output paths
    input_path = os.path.join(dicom_dir, f"{i}.dcm")
    output_path = os.path.join(output_dir, f"{i}.dcm")

    # Read the DICOM file
    dicom_data = pydicom.dcmread(input_path)

    # Check if DICOM file is valid
    if dicom_data:
        # Crop the DICOM file
        pixel_array = dicom_data.pixel_array
        cropped_pixel_array = pixel_array[bbox_y_start:bbox_y_end, bbox_x_start:bbox_x_end]

        # Resize the cropped DICOM image
        resized_image = resize(cropped_pixel_array, target_size, anti_aliasing=True, preserve_range=True)

        # Convert the resized image to uint16
        resized_image = resized_image.astype(np.uint16)

        # Update DICOM metadata
        dicom_data.Rows, dicom_data.Columns = target_size
        dicom_data.PixelData = resized_image.tobytes()

        # Save the resized DICOM file
        dicom_data.save_as(output_path)
    else:
        print(f"Error loading DICOM file: {input_path}")

print("DICOM files extracted, cropped, resized, and saved successfully.")

#crop slices

import os
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets, interact
import pydicom

# Define the directory containing DICOM files
dicom_dir = r"D:\Project_vk\Registration\CROP_DEC2"

# Load DICOM files
dicom_files = [os.path.join(dicom_dir, filename) for filename in os.listdir(dicom_dir)]

# Print the length of dicom_files
print("Number of DICOM files in dicom_dir:", len(dicom_files))

# Sort DICOM files by instance number
dicom_files.sort(key=lambda x: pydicom.dcmread(x).InstanceNumber)

# Read DICOM metadata
first_slice = pydicom.dcmread(dicom_files[0])

# Load DICOM pixel data
slices = [pydicom.dcmread(file).pixel_array for file in dicom_files]

# Convert slices to numpy array
vol = np.stack(slices)

# Display the shape of the loaded windowed volume
print("Shape of loaded CT volume (windowed):", vol.shape)

n0, n1, n2 = vol.shape

# Initialize selected_values dictionary
selected_values_sagittal = None
selected_values_coronal = None

# Interactive function to crop sagittal slice
@interact(sagittal_slice=(0, n2 - 1), 
          x_start_sagittal=widgets.IntSlider(min=0, max=n1, step=1, value=0),
          x_end_sagittal=widgets.IntSlider(min=0, max=n1, step=1, value=n1),
          y_start_sagittal=widgets.IntSlider(min=0, max=n0, step=1, value=0),
          y_end_sagittal=widgets.IntSlider(min=0, max=n0, step=1, value=n0))
def crop_sagittal_slice(sagittal_slice=264, x_start_sagittal=272, x_end_sagittal=359, y_start_sagittal=78, y_end_sagittal=326):
    global sagittal_values
    sagittal_values = {'x_start': x_start_sagittal, 'x_end': x_end_sagittal, 'y_start': y_start_sagittal, 'y_end': y_end_sagittal}
    
    cropped_slice_sagittal = vol[y_start_sagittal:y_end_sagittal, x_start_sagittal:x_end_sagittal, sagittal_slice]
    
    # Plot coronal slice
    coronal_slice = n1 // 2  # Middle slice
    x_start_coronal = 0
    x_end_coronal = n2
    cropped_slice_coronal = vol[:, x_start_coronal:x_end_coronal, coronal_slice]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    # Plot sagittal slice
    ax.imshow(cropped_slice_sagittal, cmap='gray')
    ax.axis('off')
    ax.set_title('Cropped Sagittal Slice')
    
    
    plt.show()

# Interactive function to crop coronal slice
@interact(coronal_slice=(0, n1 - 1), 
          y_start=widgets.IntSlider(min=0, max=n0, step=1, value=0),
          y_end=widgets.IntSlider(min=0, max=n0, step=1, value=n0),
          z_start=widgets.IntSlider(min=0, max=n2, step=1, value=0),
          z_end=widgets.IntSlider(min=0, max=n2, step=1, value=n2))
def crop_coronal_slice(coronal_slice=306, y_start=82, y_end=349, z_start=215, z_end=307):
    global coronal_values
    coronal_values = {'z_start': z_start, 'z_end': z_end}
    
    cropped_slice = vol[y_start:y_end, coronal_slice, z_start:z_end]
    
    # Display the cropped slice
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cropped_slice, cmap='gray')
    ax.axis('off')
    ax.set_title('Cropped Coronal Slice')
    
    plt.show()