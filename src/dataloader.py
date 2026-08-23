import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

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

class GGWPDataset(torch.utils.data.Dataset):
    """
    動態從 DataFrame 載入資料，避免一次吃光 RAM。
    每筆資料會回傳 X:(3,L) 以及 y:(5,)
    """
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.stack([
            row['Norm_Spread'],
            row['S1_Return'],
            row['S2_Return']
        ], axis=0).astype(np.float32)
        y = row[['Revert', 'Norm_Rtop', 'Norm_Top', 'Norm_Close', 'Norm_Tax']].to_numpy(dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)
