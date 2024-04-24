import torch
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset

from tqdm import tqdm

from enum import Enum

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

from hiera import hiera_base_224, Hiera
from hiera_utils import pretrained_model

from timm.models.vision_transformer_sam import Block

from decoder import FPNSegmentationHead

import os

device='cuda'
dtype=torch.float32

# Misc Region
#region
unnormalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

def imgnet_vis(img, grad=True):
    img = unnormalize(img)
    # if uses grad, detach
    if grad:
        image = img.squeeze(0).permute(1, 2, 0).detach()
    else:
        image = img.squeeze(0).permute(1, 2, 0)
    plt.imshow(torch.clip(image, 0, 1))
    plt.axis('off')
    return


def soft_jaccard_score(
    output: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)

    union = cardinality - intersection
    jaccard_score = (intersection + smooth) / (union + smooth).clamp_min(eps)
    return jaccard_score

def my_iou(pred, target):
    
    pred = F.logsigmoid(pred).exp()
    
    dims = (0, 2)
    scores = soft_jaccard_score(
            pred,
            target.type(pred.dtype),
            dims=dims,
        )
    
    
    loss = 1.0 - scores
    
    mask = target.sum(dims) > 0
    loss *= mask.float()


    return loss.mean()

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x
    
def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs

def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta**2) * tp + eps) / ((1 + beta**2) * tp + beta**2 * fn + fp + eps)

    return score
#endregion


# Dataloader Region
#region
def SVD(image, k=128):
    U, S, Vh = torch.linalg.svd(image, full_matrices=True)
    U_compressed = U[:, :, :k]
    S_compressed = S[:, :k]
    Vh_compressed = Vh[:, :k, :]
    return U_compressed @ (S_compressed[:, :, None] * Vh_compressed)
    

class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            mask_dir (string): Directory with all the masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        self.transform = transform
        self.dtype = torch.float32
        self.resize = transforms.Resize((256,256))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_names[idx])
        mask_name = os.path.join(self.mask_dir, f"mask_{self.image_names[idx]}")
        image = read_image(img_name) / 255.
        image = image.to(self.dtype)
        
        mask = read_image(mask_name).to(self.dtype)

        if self.transform:
            image = self.resize(image)
            if image.shape[0] == 1:
                image = torch.cat([image, image, image], dim=0)
            image = self.normalize(image)
            #image = SVD(image)
            mask = self.resize(mask)

        return image, mask
#endregion

# Module Region
#region

checkpoints = {
    "example_checkpoint": "https://dl.fbaipublicfiles.com/hiera/hiera_base_224.pth"
}

@pretrained_model(checkpoints, default="example_checkpoint")
def create_hiera_model():
    return Hiera(input_size=(256,256), num_classes=10)

class Model(torch.nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, depth=10):
        super().__init__()
        self.encoder = create_hiera_model()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.blocks = torch.nn.Sequential(*[
            Block(embed_dim, num_heads) for _ in range(depth)
        ])
        # self.block = Block(embed_dim, num_heads)
        self.decoder = FPNSegmentationHead(embed_dim, 1, decode_intermediate_input=False, shortcut_dims=[96,192,384,768])
        
    def forward(self, x):
        intermediates = self.encoder(x, return_intermediates=True)
        shortcuts = []
        x = intermediates[-1]
        for i in intermediates:
            shortcuts.append(i.permute(0, 3, 1, 2))
        x = self.blocks(x).permute(0, 3, 1, 2)
        x = self.decoder([x], shortcuts)

        x_binary = torch.where(x > 0.5, torch.tensor(1, device=x.device, dtype=torch.float32, requires_grad=True), torch.tensor(0, device=x.device, dtype=torch.float32, requires_grad=True))
        return x_binary

#endregion

# Training Loop
#region
def train():
    transform = transforms.Compose([transforms.Resize((256,256)),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    batch_size = 4
    root_dir = '/home/hice1/eleleux3/scratch/segment-and-replace/seg/dataset/val2017'
    image_dir = os.path.join(root_dir, 'images')
    mask_dir = os.path.join(root_dir, 'masks')

    # reuse tranforms we used earlier
    dataset = ImageMaskDataset(image_dir, mask_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = Model().to(device)
    model.train()
    optim = torch.optim.RAdam(model.parameters(), lr=0.0001)
    #mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()

    f_score_losses = []
    iou_losses = []
    losses = []
    num_epochs = 1000
    for epoch in range(num_epochs):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch+1}/{num_epochs}')
        total_loss = 0
        count = 0
        f_score_loss = 0
        iou_loss = 0
        
        for i, (sample, mask) in pbar:
            sample, mask = sample.to(device), mask.to(device)
            out = model(sample)
            mask = torch.nn.functional.interpolate(mask, scale_factor=(1/4))
            #loss = mse_loss(out, mask)
            loss = bce_loss(out, mask)
            total_loss += loss.item()
            count += 1
            loss.backward()
            optim.step()

            if epoch % 10 == 0:
                f_score_loss += f_score(out, mask, threshold=0.5)
                iou_loss += my_iou(out, mask)
                
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
            
        losses.append(total_loss / count)
        if epoch % 10 == 0:
            f_score_losses.append(f_score_loss / len(dataloader))
            iou_losses.append(iou_loss / len(dataloader))
        if epoch == 10 or (epoch % 100 == 0 and epoch != 0):
            torch.save(model.state_dict(), f"{epoch}_model3.pt")
            plt.figure(figsize=(10, 5))
            plt.plot(range(len(losses)), losses, marker='o')
            plt.title('BCE Loss per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(f'{epoch}_BCEloss_per_epoch3.png')

            plt.figure(figsize=(10, 5))
            plt.plot(range(len(f_score_losses)), f_score_losses, marker='o')
            plt.title('F_Score per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(f'{epoch}_FScore_per_epoch3.png')

            plt.figure(figsize=(10, 5))
            plt.plot(range(len(iou_losses)), iou_losses, marker='o')
            plt.title('IOU per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(f'{epoch}_IOU_per_epoch3.png')
    
    torch.save(model.state_dict(), f"final_model3.pt")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), losses, marker='o')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('loss_per_epoch3.png')

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(f_score_losses)), f_score_losses, marker='o')
    plt.title('F_Score per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f'3FScore_per_epoch3.png')

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(iou_losses)), iou_losses, marker='o')
    plt.title('IOU per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f'IOU_per_epoch3.png')

#endregion


if __name__ == "__main__":
    train()