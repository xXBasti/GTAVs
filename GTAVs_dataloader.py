import os.path
import pickle
from enum import Enum
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, check_integrity


class GTAVSSet(str,Enum):
    initial_pool = "init"
    unlabeled_pool = "unlabeled_pool"
    train_set = "train"
    val_set = "val"
    test_set = "test"
    all = "all"


class GTAVS(VisionDataset):
    """`GTAVS <https://www.cs.cit.tum.de/daml/tpl/>` Dataset.

    Args:
        root (string): Root directory of dataset where directory exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "GTA-V-Streets"
    url = "https://public.am.files.1drv.com/y4mXkigeZQ0jES9qwcZZ0CDMJioUZiU1MUeLyyg5-cCz_K40uGoNKZAnSsMFvUZgBitBsTe6OgtiNWBd-C_DMqIc3nT9p-99Y6EOlwIlfTwSHOHfaNP6uBoYlGKstoS1-ZCUTuwSTno-PypdLOgtP0HhQexLsB_RHoRJUIh-nvM5oO0Q32OC6pw0o9xexKeGeMhazHQ8_5MgIj3oOmg5ejKn9skkgt7EFHUMAErQwKUVek?AVOverride=1"
    filename = "GTA-V-Streets.zip"
    tgz_md5 = "ca58599e513873b1d001fa7d2e7013c1"
    routes = [
        ["Route1", "Route2", "Route3", "Route4", "Route5", "Route6", "Route7"],
    ]
    init_pool=["Route6"]
    unlabeled_pool=["Route2","Route4","Route5","Route7"]
    val_set=["Route1"]
    test_set=["Route3"]

    def __init__(
        self,
        root: str,
        subset: GTAVSSet = GTAVSSet.initial_pool,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        subset=self.select_set(subset)
        self.data, self.targets = self.load_data(subset)

    def load_data(self, subset):

        # now load the picked numpy arrays
        for folder_name in subset:
            file_path = os.path.join(self.root, self.base_folder, folder_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])


    def select_set(self, set):
        if set == GTAVSSet.initial_pool:
            return self.init_pool
        elif set == GTAVSSet.unlabeled_pool:
            return self.unlabeled_pool
        elif set == GTAVSSet.val_set:
            return self.init_pool + self.unlabeled_pool
        elif set == GTAVSSet.val_set:
            return self.val_set
        elif set == GTAVSSet.test_set:
            return self.test_set

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        for filename, md5 in self.routes:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not os.path.exists(fpath):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)


if __name__ == "__main__":
    GTAVS("/home/basti/Data",download=True)