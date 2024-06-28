import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets, interact
import pydicom
from skimage.transform import resize
from skimage import transform
# Load the DICOM file
dicom_path = r'D:\Project_vk\Registration\data\1027.dcm'
dicom = pydicom.dcmread(dicom_path, force=True)

xray_row_resize = 600
xray_column_resize = 600
# Extract pixel data and convert to numpy array
dicom_pixel_array = 1 - (dicom.pixel_array - dicom.pixel_array.min()) / (dicom.pixel_array.max() - dicom.pixel_array.min())  # Normalize pixel values
# Rotate the image
dicom_pixel_array = transform.rotate(dicom_pixel_array, 0)
dicom_pixel_array = resize(dicom_pixel_array, (xray_row_resize,xray_column_resize),anti_aliasing=True, preserve_range=True)
# Initialize selected_values dictionary
selected_values_xray = None

# Interactive function to crop X-ray image
@interact(x_start=widgets.IntSlider(min=0, max=dicom_pixel_array.shape[1], step=1, value=0),
          x_end=widgets.IntSlider(min=0, max=dicom_pixel_array.shape[1], step=1, value=dicom_pixel_array.shape[1]),
          y_start=widgets.IntSlider(min=0, max=dicom_pixel_array.shape[0], step=1, value=0),
          y_end=widgets.IntSlider(min=0, max=dicom_pixel_array.shape[0], step=1, value=dicom_pixel_array.shape[0]))
def crop_xray_dicom(x_start=0, x_end=dicom_pixel_array.shape[1], y_start=0, y_end=dicom_pixel_array.shape[0]):
    global xray_dicom_values
    xray_dicom_values = {'x_start': x_start, 'x_end': x_end, 'y_start': y_start, 'y_end': y_end}

    cropped_xray_dicom = dicom_pixel_array[y_start:y_end, x_start:x_end]

    # Display the cropped X-ray image
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cropped_xray_dicom, cmap='gray')
    ax.axis('off')
    ax.set_title('Cropped X-ray DICOM Image')

    plt.show()


#crop xray ADJUST HEIGHT AND WIDTH ACC TO DRR
import pydicom
import torch
import torch.nn.functional as F
import numpy as np
from skimage import transform, exposure
import pandas as pd

# Load the DICOM file
#dicom_path = r'D:\DeepLearning\dup\fin\\data\10232.dcm'
dicom_path = r'D:\Project_vk\Registration\data\1027.dcm'
dicom = pydicom.dcmread(dicom_path)

# Extract pixel data and convert to numpy array
dicom_pixel_array = 1 - dicom.pixel_array
dicom_pixel_array = resize(dicom_pixel_array, (xray_row_resize,xray_column_resize),anti_aliasing=True, preserve_range=True)

# Assuming it's RGB DICOM, split into channels
dicom_pixel_array = dicom_pixel_array[:, :,1]
dicom_pixel_array = dicom_pixel_array.astype(np.float32)  # Convert to float32

# Rotate the image
rotated_image = transform.rotate(dicom_pixel_array, 0)


# Extract bounding box values
bbox_x = xray_dicom_values['x_start']
bbox_y = xray_dicom_values['y_start']
bbox_width = xray_dicom_values['x_end']
bbox_height = xray_dicom_values['y_end']

# Crop the rotated image based on bounding box values
cropped_image = rotated_image[bbox_y:bbox_height,bbox_x:bbox_width]
cropped_image = resize(cropped_image, (40,40),anti_aliasing=True, preserve_range=True)

# Convert the cropped image from NumPy array to PyTorch tensor
cropped_tensor = torch.tensor(cropped_image)

# Clamp the intensity values of the cropped tensor from 300 to 1500
clipped_tensor = torch.clamp(cropped_tensor, 0, 65000)



# Reshape the tensor to have batch dimension of 1 and channel dimension of 1
clipped_tensor_reshaped = clipped_tensor.unsqueeze(0).unsqueeze(0)

# Resize the image using bilinear interpolation to the desired shape
desired_shape = (HEIGHT, WIDTH)
resized_clipped_tensor = F.interpolate(clipped_tensor_reshaped, size=desired_shape, mode='bilinear', align_corners=False)


# Ensure the shape is torch.Size([1, 1, 200, 120])
xray = resized_clipped_tensor.view(1, 1, HEIGHT, WIDTH)





