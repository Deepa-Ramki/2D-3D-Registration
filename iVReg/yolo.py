import os

def train_yolo():
    os.system("python train.py --img 640 --batch 16 --epochs 400 --data coronal.yaml --weights yolov5s.pt --cache")

def detect_yolo():
    os.system("python detect.py --weights coronal.pt --img 640 --conf 0.25 --source data/coronal_images --classes 8")

if __name__ == "__main__":
    # Train the YOLO model
    train_yolo()
    
    # Detect using the YOLO model
    detect_yolo()


# Model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, etc.
model = torch.hub.load('ultralytics/yolov5', 'custom', 'coronal.pt')  # custom trained model

# Images
im = '/home/htic/Project_vk/yolov5/data/images/1.jpg'  # or file, Path, URL, PIL, OpenCV, numpy, list

# Inference
results = model(im)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

#results.xyxy[0]  # im predictions (tensor)
#results.pandas().xyxy[0]  # im predictions (pandas)


# Model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, etc.
model = torch.hub.load('ultralytics/yolov5', 'custom', 'coronal.pt')  # custom trained model

# Images
im = '/home/htic/Project_vk/yolov5/data/images/1.jpg'  # or file, Path, URL, PIL, OpenCV, numpy, list

# Inference
results = model(im)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

#results.xyxy[0]  # im predictions (tensor)
#results.pandas().xyxy[0]  # im predictions (pandas)


def extract_values_by_column5(input_value, tensor):
    # Find the row index where the input_value is located in column 5
    row_index = (tensor[:, 5] == input_value).nonzero(as_tuple=True)[0]

    if len(row_index) > 0:
        # Extract columns 0 to 3 from the corresponding row
        extracted_values = tensor[row_index, :4]
        return extracted_values.squeeze().tolist()
    else:
        return None  # Return None if the input value is not found

# Example usage
input_value = 3
output = extract_values_by_column5(input_value, results.xyxy[0])

if output:
    print(f"For input value {input_value}, the extracted values are: {output}")
else:
    print(f"Input value {input_value} not found.")



# Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, etc.
model = torch.hub.load('ultralytics/yolov5', 'custom', 'sagittal.pt')  # custom trained model

# Images
im = '/home/htic/Project_vk/yolov5/data/sagittal_images/5.jpg'  # or file, Path, URL, PIL, OpenCV, numpy, list

# Inference
results = model(im)

# Convert results to pandas DataFrame
df = results.pandas().xyxy[0]

# Save results to CSV
csv_file_path = '/home/htic/Project_vk/yolov5/data/sagittal_images/results.csv'
df.to_csv(csv_file_path, index=False)

print(f"Results saved to {csv_file_path}")