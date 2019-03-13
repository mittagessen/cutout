import glob
import torch
import torch.utils.data as data

from PIL import Image
from pathlib import Path
from torchvision import transforms


default_transforms = transforms.Compose([transforms.Resize((900,900)), transforms.Lambda(lambda x: x.convert('RGB')), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class CutoutDataset(data.Dataset):
    """
    A data loader where the samples are arrange in this way:

    root/F/abc.png
    root/F/abd.png
    root/F/abe.png
    ...
    root/J/123.png
    root/J/234.png
    root/J/567.png

    where F and J are float values between 0 and 1.

    Args:
        root (str): Root directory path
    """

    def __init__(self, root, transform=default_transforms):
        self.root = root
        self.transform = transform
        ims = glob.glob('{}/*/*.png'.format(root))
        self.samples = []
        for im in ims:
            self.samples.append((im, float(Path(im).parts[-2])))

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, torch.tensor(target)

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
