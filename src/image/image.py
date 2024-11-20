from viam.media.video import CameraMimeType, ViamImage
from typing import Dict, List
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import io


def get_tensor_from_np_array(np_array: np.ndarray) -> torch.Tensor:
    """
    returns an RGB tensor
    """
    uint8_tensor = torch.from_numpy(np_array).permute(2, 0, 1).contiguous()
    float32_tensor = uint8_tensor.to(dtype=torch.float32)
    return uint8_tensor, float32_tensor


class ImageObject:
    def __init__(self, viam_image: ViamImage, device=None):
        # self.encodings: Dict[str, CameraMimeType] = {}
        # self.np_array: np.ndarray= None
        # self.supported_mime_types = ["image/jpeg", "image/png"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pil_image = Image.open(io.BytesIO(viam_image.data)).convert("RGB")
        self.np_array = np.array(self.pil_image, dtype=np.uint8)
        self.uint8_tensor, self.float32_tensor = get_tensor_from_np_array(self.np_array)
        self.uint8_tensor.to(device)
        self.float32_tensor.to(device)

    # def add_encoding(mime_type: CameraMimeType, bytes):
    #     self.encodings[mime_type] = bytes

    # def get_cropped_tensor():
    #     pass

    # def get_viam_image():
    #     pass

    # def get_np_array(self):
    #     if self.np_array is not None:

    #     if self.encodings:

    # def get_tensor(self):
    #     if self.tensor is not None:
    #         return self.tensor
    #     else:

    # def get_pil_image(self):
    #     if self.pil_image is not None:
    #         return
    #     return Image.open(io.BytesIO(image_bytes)).convert("RGB")
