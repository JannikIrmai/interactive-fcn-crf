import os
from torch.utils.data import DataLoader
import numpy as np
from dataset import TestDataset, TrainDataset


CSI_LABELS = [
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
    'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
    'L1', 'L2', 'L3', 'L4', 'L5', 'S1', 'S2'
]

VERSE_LABELS = [
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
    'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
    'L1', 'L2', 'L3', 'L4', 'L5', 'L6',
    'S1', 'S2',
    'T13'
]


def get_data_iter(data_dir: str, train_dir: str, test_dir: str, val_size: int = 40, resolution: int = 2, batch_size: int = 2, seed: int = None,
                  sigma: float = 3, num_workers: int = 4):

    paths = {}

    # -- load paths
    tr_dir = os.path.join(data_dir, train_dir)
    paths['train'] = [os.path.join(tr_dir, x) for x in os.listdir(tr_dir) if
                      f'_{resolution}mm_3d' in x]
    te_dir = os.path.join(data_dir, test_dir)
    paths['test'] = [os.path.join(te_dir, x) for x in os.listdir(te_dir) if
                     f'_{resolution}mm_3d' in x]

    # -- split the train paths into val paths
    paths['train'] = np.random.RandomState(seed).permutation(paths['train'])
    paths['valid'] = paths['train'][-val_size:]
    paths['train'] = paths['train'][:-val_size]

    print('data size:')
    print('train:', len(paths['train']))
    print('valid:', len(paths['valid']))
    print('test :', len(paths['test']))

    # -- prepare data loaders
    def get_data_loader(dataset_class, paths, batch_size, shuffle=True, collate_fn=None):

        return DataLoader(
            dataset_class(paths, sigma),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=max(0, num_workers),
        )

    data_iter = {
        'train': get_data_loader(TrainDataset, paths['train'], batch_size, shuffle=True,
                                 collate_fn=TrainDataset.collate_fn),
        'valid': get_data_loader(TestDataset, paths['valid'], 1, shuffle=False),
        'test': get_data_loader(TestDataset, paths['test'], 1, shuffle=False),
        'train_forInference': get_data_loader(TestDataset, paths['train'], 1, shuffle=False),
    }

    return data_iter
