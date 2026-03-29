import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
import torch
import torchvision.transforms as transforms

resizer_collection = {
    "nearest": InterpolationMode.NEAREST,
    "box": InterpolationMode.BOX,
    "bilinear": InterpolationMode.BILINEAR,
    "hamming": InterpolationMode.HAMMING,
    "bicubic": InterpolationMode.BICUBIC,
    "lanczos": InterpolationMode.LANCZOS}

class PairedDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        mode,
        resize_size=None,
        resizer=None,
        random_flip=False,
        normalize=True,
    ):
        super().__init__()
        self.microscopy_modalities = {
            "Phase": "phase constrast",
            "Opti": "optical",
            "fluo": "fluorescence",
        }
        self.data_dir = dataset_dir
        self.mode = mode
        self.normalize = normalize
        self.trsf_list = []
        self.img2tensor = transforms.ToTensor()
        if resize_size:
            if resizer:
                self.trsf_list.append(
                    transforms.Resize(resize_size, interpolation=resizer_collection[resizer])
                )
            else:
                self.trsf_list.append(transforms.RandomCrop(resize_size))
        if self.mode == "train" and random_flip:
            self.trsf_list.append(transforms.RandomHorizontalFlip(p=random_flip))
        self.trsf = transforms.Compose(self.trsf_list)
        self.load_dataset()

    def load_dataset(self):
        root = os.path.join(self.data_dir, self.mode)
        self.data = ImageFolder(root=root)
        self.mask_data = ImageFolder(root=os.path.join(self.data_dir, "grayscale_v3", self.mode))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        txt_prompt = self.get_prompt(index)
        img, _ = self.data[index]
        mask, _ = self.mask_data[index]
        img = self.img2tensor(img)
        mask = self.img2tensor(mask)
        full_stack = torch.cat((img, mask), 0)
        if self.normalize:
            img = img * 2.0 - 1
            mask = mask * 2.0 - 1
        return {
            "image": self.trsf(img),
            "mask": self.trsf(mask),
            "caption": txt_prompt,
            "file_name": self.data.imgs[index][0].split("/")[-1],
        }

    def get_prompt(self, index):
        full_path, _ = self.data.imgs[index]
        img_class = full_path.split("/")[-2]
        file_name = full_path.split("/")[-1]
        file_metadata = file_name.split("_")
        img_modality = file_metadata[1]
        time_string = file_metadata[4]
        days = int(time_string.split("d")[0])
        hours = int(time_string.split("d")[-1].split("h")[0])
        full_time = (days * 24) + hours
        out_txt = f"{img_class} {self.microscopy_modalities[img_modality]} microscopy image at {full_time} hours of growth"
        return out_txt