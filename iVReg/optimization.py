from diffdrr.drr import DRR, Registration
from diffdrr.metrics import NormalizedCrossCorrelation2d,Sobel

criterion = NormalizedCrossCorrelation2d()
from skimage.transform import resize
import torch

# Check if img is on CUDA and move it to CPU if necessary
if img.is_cuda:
    img = img.cpu()

# Convert normalized_image and img tensors to Double data type
#normalized_image = normalized_image.float()

# Now you can use img for comparison
print(criterion(output.cpu(), img).item())

import pandas as pd
from tqdm import tqdm

def optimize(
    reg: Registration,
    img,
    lr_rotations=5.3e-2,
    lr_translations=7.5e1,
    momentum=0,
    dampening=0,
    n_itrs=250,
    optimizer="sgd",  # 'sgd' or `adam`
):
    criterion = NormalizedCrossCorrelation2d()
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(
            [
                {"params": [reg.rotation], "lr": lr_rotations},
                {"params": [reg.translation], "lr": lr_translations},
            ],
            momentum=momentum,
            dampening=dampening,
            maximize=True,
        )
    else:
        optimizer = torch.optim.Adam(
            [
                {"params": [reg.rotation], "lr": lr_rotations},
                {"params": [reg.translation], "lr": lr_translations},
            ],
            maximize=True,
        )

    params = []
    losses = []
    for itr in tqdm(range(n_itrs), ncols=50):
        # Save the current set of parameters
        alpha, beta, gamma = reg.get_rotation().squeeze().tolist()
        bx, by, bz = reg.get_translation().squeeze().tolist()
        params.append([i for i in [alpha, beta, gamma, bx, by, bz]])

        # Run the optimization loop
        optimizer.zero_grad()
        estimate = reg()
        estimate = estimate.float()
        img = img.float()
        loss = criterion(img, estimate)
        loss.backward(retain_graph=True)
        optimizer.step()
        losses.append(loss.item())

        if loss > 0.95:
            tqdm.write(f"Converged in {itr} iterations")
            break

    df = pd.DataFrame(params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
    df["loss"] = losses
    
    # Extract the max loss and its corresponding parameters
    max_loss_index = df['loss'].idxmax()
    max_loss_params = df.iloc[max_loss_index]
    
    return df, max_loss_params

baser_ms = rotation.clone()
baset_ms = translation.clone()
# Base SGD
# Ensure resized_dicom_tensor is on the same device as other tensors
normalized_image = output.to(device)
drr = DRR(volume, spacing, sdr=474.6, height=HEIGHT,width=WIDTH, delx=DELX).to(device)
reg = Registration(
    drr,
    baser,
    baset,
    parameterization="euler_angles",
    convention="ZYX",
)
params_base_ms,max_loss_params_base_ms = optimize_ms(reg, normalized_image)
del drr
print("Max Loss Parameters:")
print(max_loss_params_base_ms)

baser_g = rotation.clone()
baset_g = translation.clone()


momentr_ms = rotation.clone()
momentt_ms = translation.clone()

# Base SGD
# Ensure resized_dicom_tensor is on the same device as other tensors
normalized_image = output.to(device)
drr = DRR(volume, spacing, sdr=474.6, height=HEIGHT,width=WIDTH, delx=DELX).to(device)
reg = Registration(
    drr,
    momentr_ms,
    momentt_ms,
    parameterization="euler_angles",
    convention="ZYX",
)
params_momentum_ms,max_loss_params_momentum_ms = optimize_ms(reg, normalized_image,momentum=0.9)
del drr
print("Max Loss Parameters:")
print(max_loss_params_momentum_ms)


moment_dampemnr_ms = rotation.clone()
moment_dampemnt_ms = translation.clone()

# Base SGD
# Ensure resized_dicom_tensor is on the same device as other tensors
normalized_image = output.to(device)
drr = DRR(volume, spacing, sdr=474.6, height=HEIGHT,width=WIDTH, delx=DELX).to(device)
reg = Registration(
    drr,
    moment_dampemnr_ms,
    moment_dampemnt_ms,
    parameterization="euler_angles",
    convention="ZYX",
)
params_momentum_dampemn_ms,max_loss_params_momentum_dampemn_ms = optimize_ms(reg, normalized_image,momentum=0.9, dampening=0.1)
del drr
print("Max Loss Parameters:")
print(max_loss_params_momentum_dampemn_ms)



drr = DRR(volume, spacing, sdr=474.6, height=HEIGHT, width=WIDTH, delx=DELX).to(device)
reg = Registration(
    drr,
    r_1,
    t_1,
    parameterization="euler_angles",
    convention="ZYX",
)
params_adam_ms,max_loss_params_ms = optimize_ms(reg, normalized_image, 0.01, 0.1, optimizer="adam")
del drr

# Reset the environment variable after running the script if needed
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ''


print("Max Loss Parameters:")
print(max_loss_params_ms)