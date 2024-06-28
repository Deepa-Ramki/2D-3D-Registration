import os
import numpy as np
import pydicom
import torch
import matplotlib.pyplot as plt
from diffdrr.drr import DRR
from diffdrr.visualization import plot_drr 


dicom_dir = "D:\Project_vk\Registration\perfect"

# Load DICOM files
dicom_files = [os.path.join(dicom_dir, filename) for filename in os.listdir(dicom_dir)]

# Sort DICOM files by instance number
dicom_files.sort(key=lambda x: pydicom.dcmread(x).InstanceNumber)

# Read DICOM metadata
first_slice = pydicom.dcmread(dicom_files[0])
spacing = [0.703125, 0.703125, 0.625]  # Example spacing values
#spacing =[0.519531, 0.519531, 0.625]#murr
#slice_thickness = first_slice.SliceThickness
#spacing = first_slice.PixelSpacing

# Load DICOM pixel data and clip intensities
clip_min = -150
clip_max = 4500

slices = [np.clip(pydicom.dcmread(file).pixel_array, clip_min, clip_max) for file in dicom_files]

# Convert slices to numpy array
volume = np.stack(slices)
volume = np.array(volume, dtype=np.ndarray)
volume = volume.astype(np.float32)

# Display information about the loaded volume
print(f"Volume shape: {volume.shape}")
print(f"Spacing: {spacing}")

# Calculate bounding box dimensions (assuming equal spacing between slices)
#bx, by, bz = [49.0, 45.0, 40.0]  # Example bounding box dimensions
bx, by, bz = np.array(volume.shape) * np.array(spacing) / 2
print(bx, by, bz)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

torch.cuda.empty_cache()

#slices = [pydicom.dcmread(file).pixel_array for file in dicom_files]

X = "sagittal"
if X == "sagittal":
    WIDTH = 55
    pitch_angle = 90
else:
    WIDTH = 75
    pitch_angle = 180


HEIGHT = 90
WIDTH = 50
DELX = 1.5

#PERFORM DRR
yaw_angle = 90  # Adjust the yaw angle
#pitch_angle = 180 # Adjust the pitch angle
roll_angle = 0
yaw_angle_rad = np.radians(yaw_angle)
pitch_angle_rad = np.radians(pitch_angle)
roll_angle_rad = np.radians(roll_angle)# Adjust the roll angle
# Create a new rotation tensor with the desired angles
rotation = torch.tensor([[yaw_angle_rad, pitch_angle_rad, roll_angle_rad]], device=device)
translation = torch.tensor([[bx, by, bz]], device=device)
rotation = rotation.float()

drr = DRR(
    volume,      # The CT volume as a numpy array
    spacing,     # Voxel dimensions of the CT
    sdr=474.6,   # Source-to-detector radius (half of the source-to-detector distance)
    height=HEIGHT,
    width=WIDTH,  # Height of the DRR (if width is not seperately provided, the generated image is square)
    delx=DELX,    # Pixel spacing (in mm)
).to(device)

# Set the camera pose with rotation (yaw, pitch, roll) and translation (x, y, z)
#rotation = torch.tensor([[torch.pi, 0.0, torch.pi / 2]], device=device)
translation = torch.tensor([[bx, by, bz]], device=device)
translation = translation.float()

# 📸 Also note that DiffDRR can take many representations of SO(3) 📸
# For example, quaternions, rotation matrix, axis-angle, etc...
img = drr(rotation, translation, parameterization="euler_angles", convention="ZYX")
plot_drr(img, ticks=False)
plt.show()