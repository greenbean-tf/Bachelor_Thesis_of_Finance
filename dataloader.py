import torch
from torch.utils.data import Dataset
from torchvision import transforms

def add_noise(sequence, noise_level=0.25):
    noise = torch.randn_like(sequence) * torch.unsqueeze(torch.std(sequence,dim=1),1)*noise_level
    return sequence + noise

def time_mask(sequence, mask_ratio=0.5):
    mask = torch.bernoulli(torch.full(sequence.shape, mask_ratio))
    return sequence * (1 - mask)

def random_flipping(sequence, flip_ratio=0.5):
    if torch.rand(1) < flip_ratio:
        return sequence
    else:
        return -sequence

def random_swap(sequence, swap_ratio=0.5):
    if torch.rand(1) < swap_ratio:
        return sequence
    else:
        sequence[[1, 2], :] = sequence[[2, 1], :]
        return sequence


class Dataset_PairsTrade(Dataset):
    """
        Args:
            transform       : Transformation to your dataset
    """

    def __init__(self, data, labels, transform):
        super().__init__()
        self.data = data.astype('float32')
        if labels is not None:
            self.labels = labels.astype('float32')
            self.has_labels = True
        else:
            self.has_labels = False
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.has_labels:
            return self.transform(self.data[index]), self.labels[index]
        else:
            return self.transform(self.data[index])


def train_transform(sequence):
    sequence = transforms.ToTensor()(sequence)
    sequence = torch.squeeze(sequence)
    #sequence = add_noise(sequence)
    #sequence = time_mask(sequence)
    #sequence = random_flipping(sequence)
    sequence = random_swap(sequence)
    return sequence

def eval_transform(sequence):
    sequence = transforms.ToTensor()(sequence)
    sequence = torch.squeeze(sequence)
    return sequence


