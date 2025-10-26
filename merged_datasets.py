from bhaang.dataset.medmnist.read_data import pytorch_dataset
from bhaang.Medical_imaging.logger import CSVLogger, TextLogger
import torch


def merge_datasets(name, splits=["train", "val", "test"], download=False, as_rgb=False, size=28, transform=None, target_transform=None):
    """
    Merge datasets from the specified splits into a single dataset.
    
    Args:
        name (str): The name of the dataset to merge.
        splits (list): List of splits to merge. Default is ["train", "val", "test"].
        
    Returns:
        Dataset: Merged dataset containing data from the specified splits.
    """
    datasets = []
    for split in splits:
        dataset = pytorch_dataset(split=split, data_flag=name, download=download, as_rgb=as_rgb, size=size, transform=transform, target_transform=target_transform)
        datasets.append(dataset)
    merged_dataset = torch.utils.data.ConcatDataset(datasets)
    return merged_dataset
    