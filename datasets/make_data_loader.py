from torch.utils.data import DataLoader
import datasets
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist

# collate_utils.py
import torch

def make_data_loader(spec, tag, log, num_workers=4):
    if spec is None:
        return None
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        shuffle=(tag == 'train'), num_workers=num_workers, persistent_workers = True, prefetch_factor = 2, pin_memory=True)
    return loader

def make_data_loader_ddp(spec, tag, log, num_workers=4):
    if spec is None:
        return None
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))

    sampler = DistributedSampler(dataset, shuffle=(tag == 'train'))
    loader = DataLoader(dataset,
                    batch_size=spec['batch_size'],
                    sampler=sampler,
                    shuffle=(sampler is None and tag == 'train'),
                    num_workers=num_workers,
                    pin_memory=True)    

    return loader

def make_data_loaders_base(datasets_class, tags, log, ddp=False, num_workers=4):
    if ddp is False:
        loader = make_data_loader(datasets_class, tags, log, num_workers=4)
    else:
        loader = make_data_loader_ddp(datasets_class, tags, log, num_workers=num_workers)
    return loader

# def make_data_loaders():
#     train_loader = make_data_loader(config.get('train_dataset'), tag='train')
#     val_loader = make_data_loader(config.get('val_dataset'), tag='val')
#     return train_loader, val_loader