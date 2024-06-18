<h1 align="center">2D/3D Fluoro-CT Spine Registration</h1>

<p  align="center">  
  
   In this study, we propose a comprehensive methodology for surgical planning and navigation by integrating 2D and 3D imaging techniques. The methodology includes four key processes: `DRR generation`, `neural style transfer`, and registration between DRRs and X-ray images. CT volumes of a spine phantom were processed using YOLO-V5 models to crop regions of interest and generate DRRs. Neural style transfer addressed stylistic differences between DRRs and X-rays while preserving content. Registration using rigid transformation aligned DRRs with X-rays, with gradient-normalized cross-correlation as the metric and optimization techniques like `stochastic gradient descent with momentum` for convergence.
</p>

<h3 > <i>Index Terms</i> </h3> 

 :diamond_shape_with_a_dot_inside:2D/3D Registration
  :diamond_shape_with_a_dot_inside: Surgical planning
  :diamond_shape_with_a_dot_inside:Digitally Reconstructed Radiographs
  :diamond_shape_with_a_dot_inside:Style Transfer
  :diamond_shape_with_a_dot_inside:YOLOv5
</div>


## <div align="center">Getting Started</div>

<details>
  <summary><i>System Workflow</i></summary>

  
The process starts with acquiring X-ray images of the patient's spine from two angles: `anterior-posterior (AP)` and `lateral-posterior (LP)`

**Image Preprocessing**:These images are pre-processed and augmented to improve their quality and diversity for training the model
<br/>

**Model Training**:</ln> The processed images are used to train a deep learning model to automatically segment (identify) vertebrae in X-ray images.
<br/>

**Surgical Planning**:
    - Surgeons upload new AP and LP X-ray images for planning.
    - The GUI displays these images, allowing surgeons to select the vertebra of interest for screw placement.
    - Utilizing vertebra segmentation results, the GUI automatically adds screws to the corresponding vertebral bounding boxes.
    - Surgeons can then simulate and adjust screw placement on the patient's spine within the GUI.
    - Once satisfied, the GUI generates a surgical plan detailing screw size, type, and location.
<br/>


</details>
<details open>
<summary><i>Vertebrae Segmentation with Improved YOLOv5</i></summary>

  
**Image Input**: The system starts with AP and LP X-ray images for vertebrae segmentation.

<br/>
<div align="center">
<div style="display: flex; flex-direction: row;">
    <img class="img"src="Figure_commonmark/testSET07AP.jpg" width="300">
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <img class="img"src="Figure_commonmark/testSET07AP.jpg" width="300"> 
</div>
</div>
 <p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   :small_orange_diamond: Fig 1: Anterior-Posterior (AP) of Spine 
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   :small_orange_diamond: Fig 2: Lateral-Posterior (LP) of Spine</p>
<br/>

**Spine-Vision Model**: These images are fed into a custom deep learning model called "Spine-Vision" for segmentation.This model is based on the `YOLOv5` architecture with two key improvements:
        -  `Attention Feature (AF)` module: This module helps the model learn complex features from spine X-rays, particularly important due to variations in vertebrae shape and size.
        - `Channel Attention (CA)` module: This module helps improve object localization and mitigates the slight decrease in confidence scores caused by the AF module.
    - The model outputs bounding boxes for each vertebra in the images.
    <br/>
    
</details>

<details open>
<summary><i>Pedicle Screw Placement and Planning</i></summary>
  
**Screw Visualization**: Pedicle screws are visualized as 3D cylinders with X, Y, and Z dimensions. However, they are displayed as 2D circles on the AP and LP images.
<p align="center">
  <img src="Figure_commonmark/image.png" width ="600" height ="400" >
</p>

<div align = "center">
  
  :small_orange_diamond: Fig 3: Optimal positioning and strategic arrangement of screws on AP and LP images</span>
</div>

<br/>

 **Planning Process:**
  Surgeons label the vertebra of interest.The system automatically segments the vertebra using the Spine-Vision model. Based on the segmentation results, a screw is automatically positioned within the vertebra's bounding box. Surgeons can then adjust screw placement in either the AP or LP image. Any adjustments are automatically reflected in the corresponding image due to the shared 3D representation of the screw. The screw is defined by two points: `Entry point`: The location on the spine where the surgeon makes an incision for screw insertion. `Target point` : The desired endpoint within the vertebra for screw placement.
</details>

## <div align="center">Methodology</div>

<p align="center">
  <img src=Figure_commonmark/Screenshot%202024-06-04%20161929.png>
</p>
<div align = "center">
  
  :small_orange_diamond: Fig 4:Block diagram of proposed work: Graphical user interface (GUI) using Vertebrae  3D segmentation
</div>

## <div align="center">Pre-requisites</div>
Before installing and running the project, ensure you have the following prerequisites:

 :grey_exclamation: Download and install Jupyter Notebook from the `Jupyter website`.
 
 :wavy_dash: The version for this project is **Jupyter Notebook 6.0.3**.
  
  :grey_exclamation: YOLOv5 

 
 :wavy_dash:  See the [YOLOv5 Docs](https://docs.ultralytics.com/yolov5) for full documentation on training, testing and deployment. See below for quickstart examples.
 
:grey_exclamation: DiffDRR

:wavy_dash: `DiffDRR` is a PyTorch-based digitally reconstructed radiograph (DRR) generator that provides

:small_orange_diamond:Differentiable X-ray rendering
:small_orange_diamond:GPU-accelerated synthesis and optimization
:small_orange_diamond:A pure Python implementation

To include DiffDRR in your project, follow these steps:

1. Download from the official [DiffDRR GitHub repository](https://github.com/eigenvivek/DiffDRR/blob/main/diffdrr/pose.py):

    ```bash
    git clone https://github.com/eigenvivek/DiffDRR/blob/main/diffdrr/pose.py
    cd eigen-git-mirror
  
    ```

2. Locate the `pose.py` file in the `diffdrr` directory:

    ```bash
    cd diffdrr
    ls pose.py
    ```

3. Include the DiffDRR directory in your project's include path. 


## <div align="center">Installation</div>
:arrow_right:Clone the Repository
```bash
git clone https://github.com/yourusername/2D-3D-Registration.git
```

:arrow_right:Navigate to the Project Directory
```bash
cd 2D-3D-Registration
```
:arrow_right:Install Dependencies
```bash
pip install -r requirements.txt
```
## <div align="center">Environments</div>
<div align="center">
  <a href="https://jupyter.org/">
    <img src="https://jupyter.org/assets/homepage/main-logo.svg" width="10%" alt="Jupyter Notebook" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="5%" alt="" />
  <a href="https://bit.ly/yolov5-paperspace-notebook">
    <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-gradient.png" width="10%" alt="YOLOv5" /></a>
</div>
