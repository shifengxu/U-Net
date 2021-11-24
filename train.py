import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-6
GPU_IDS = [3]
DEVICE = f"cuda:{GPU_IDS[0]}" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
NUM_EPOCHS = 200
NUM_WORKERS = 6
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
# TRAIN_IMG_DIR = "data/train_images/"
# TRAIN_MSK_DIR = "data/train_masks/"
# VAL_IMG_DIR = "data/val_images/"
# VAL_MSK_DIR = "data/val_masks/"
TRAIN_IMG_DIR = "data_pascal/datasets/imgs/pascal/"
TRAIN_MSK_DIR = "data_pascal/datasets/masks/pascal/"
VAL_IMG_DIR = "data_pascal/datasets_val/imgs/pascal/"
VAL_MSK_DIR = "data_pascal/datasets_val/masks/pascal/"

def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        # data shape is [16, 3, 160, 240], and target shape is [16, 160, 240]. So need to unsqueeze target.
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data) # predictions shape is [16, 1, 160, 240]
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        # sets the gradients of all optimized torch.Tensor to zero
        # by default, PyTorch will do gradient accumulation. Here clear it manually.

        loss.backward()
        # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True
        # These are accumulated into x.grad for every parameter x. In pseudo-code
        # x.grad += dloss/dx

        optimizer.step()
        # optimizer.step() updates the value of x using the gradient x.grad. SGD optimizer performs:
        # x += -lr * x.grad

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    # Albumentations ensures that the input image and the output mask will
    # receive the same set of augmentations with the same parameters.
    train_transform = A.Compose( # create an instance of Compose class.
        [ # define a list of augmentations
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize( # img = (img - mean * max_pixel_value) / (std * max_pixel_value)
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(), # it is a class. To convert image and mask to torch.Tensor
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1)  # move and/or cast the parameters and buffers
    if len(GPU_IDS) > 1:
        model = torch.nn.DataParallel(model, device_ids=GPU_IDS)
    model = model.to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # model.parameters(): returns an iterator over module parameters. Typically passed to an optimizer

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MSK_DIR,
        VAL_IMG_DIR,
        VAL_MSK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    checkpoint_fname = "my_checkpoint.pth.tar"
    if LOAD_MODEL:
        load_checkpoint(checkpoint_fname, model)


    check_accuracy(val_loader, model, device=DEVICE)

    for epoch in range(NUM_EPOCHS):
        print(f"E({epoch}/{NUM_EPOCHS})...")
        train_fn(train_loader, model, optimizer, loss_fn)

        if epoch == NUM_EPOCHS - 1 or epoch > 0 and epoch % 10 == 0:
            save_checkpoint({"state_dict": model.state_dict()}, checkpoint_fname)

        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_ori = epoch == 0
        save_predictions_as_imgs(val_loader, model, epoch, save_ori, "saved_images/", device=DEVICE)
    # for


if __name__ == "__main__":
    main()
