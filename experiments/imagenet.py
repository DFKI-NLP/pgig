from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
import torchvision.datasets.imagenet


def get_imagenet_dataloader(data_dir, transforms, split_size, batch_size, imagenet_download_key):
    download=False
    if imagenet_download_key is not None:
        download=True
        # Imagenet requires an account, hence this hack
        archive_dict = torchvision.datasets.imagenet.ARCHIVE_DICT
        for k in archive_dict.keys():
            archive_dict[k]['url'] = archive_dict[k]['url'].replace('nnoupb', imagenet_download_key)

    dataset = torchvision.datasets.ImageNet(data_dir, split='val', download=download, transform=transforms)

    if split_size < len(dataset):
        indices = stratified_shuffle_indices(dataset.targets, split_size)
        dataset = Subset(dataset, indices)

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dl


def stratified_shuffle_indices(labels, size):
    inputs = list(range(len(labels)))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=size, random_state=0)
    return next(sss.split(X=inputs, y=labels))[1]
