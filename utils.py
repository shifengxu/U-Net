import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename):
    print(f"=> Saving checkpoint: {filename}")
    torch.save(state, filename)

def load_checkpoint(filename, model):
    print(f"=> Loading checkpoint: {filename}")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers, # how many subprocesses to use for data loading. default 0.
        pin_memory=pin_memory, # the data loader will copy Tensors into CUDA pinned memory before return them.
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    model.eval() # set the module in evaluation mode. equivalent with model.train(False)

    with torch.no_grad(): # context-manager that disable gradient calculation
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}%")
    model.train() # set the module in training mode

def save_predictions_as_imgs(loader, model, epoch, save_ori=True, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}e{epoch:03}_b{idx}_pred.png")
        if save_ori:
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}e{epoch:03}_b{idx}_mask.png")
            torchvision.utils.save_image(x, f"{folder}e{epoch:03}_b{idx}_orig.jpg")
    model.train()