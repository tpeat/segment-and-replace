import torch
import torch.nn.functional as F
from hiera import hiera_base_224, Hiera
from hiera_utils import pretrained_model
from decoder import FPNSegmentationHead

checkpoints = {
    "example_checkpoint": "https://dl.fbaipublicfiles.com/hiera/hiera_base_224.pth"
}

@pretrained_model(checkpoints, default="example_checkpoint")
def create_hiera_model():
    return Hiera(input_size=(256,256), num_classes=10)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = create_hiera_model()
        self.decoder = FPNSegmentationHead(768, 1, decode_intermediate_input=False, shortcut_dims=[96,192,384,768])
        
    def forward(self, x):
        intermediates = self.encoder(x, return_intermediates=True)
        shortcuts = []
        for i in intermediates:
            shortcuts.append(i.permute(0, 3, 1, 2))
        x = self.decoder([shortcuts[-1]], shortcuts)
        return x