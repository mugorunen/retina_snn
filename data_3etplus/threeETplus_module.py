import argparse, json, yaml, os

import torch
import pdb
from torch.utils.data import DataLoader
import tonic.transforms as transforms
from data_3etplus.ThreeET_plus import ThreeETplus_Eyetracking
from data_3etplus.custom_transforms import ScaleLabel, NormalizeLabel, \
    TemporalSubsample, NormalizeLabel, SliceLongEventsToShort, \
    EventSlicesToMap, SliceByTimeEventsTargets, \
    Jitter

from tonic import SlicedDataset, DiskCachedDataset
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from utils.custom_visualizations import imshow3d

def get_threeetplus_dataloader(device):
    with open(os.path.join("/home/muhammed/Desktop/retina_snn/configs/", "sliced_baseline.json"), 'r') as f:
        # print(f"Loading config from {config_file}")
        config = json.load(f)

    # We can access configuration values using a dot
    args = argparse.Namespace(**config)
    
    factor = args.spatial_factor # spatial downsample factor
    temp_subsample_factor = args.temporal_subsample_factor # downsampling original 100Hz label to 20Hz
    
    # First we define the label transformations
    label_transform = transforms.Compose([
        ScaleLabel(factor),
        TemporalSubsample(temp_subsample_factor),
        NormalizeLabel(pseudo_width=640*factor, pseudo_height=480*factor)
    ])

    # Then we define the raw event recording and label dataset, the raw events spatial coordinates are also downsampled
    train_data_orig = ThreeETplus_Eyetracking(save_to=args.data_dir, split="train", \
                    transform=transforms.Downsample(spatial_factor=factor), 
                    target_transform=label_transform, dataset=args.dataset)
    val_data_orig = ThreeETplus_Eyetracking(save_to=args.data_dir, split="val", \
                    transform=transforms.Downsample(spatial_factor=factor),
                    target_transform=label_transform, dataset=args.dataset)
    
    slicing_time_window = args.train_length*int(10000/temp_subsample_factor) #microseconds
    train_stride_time = int(10000/temp_subsample_factor*args.train_stride) #microseconds
    valid_stride_time = int(10000/temp_subsample_factor*args.val_stride) #microseconds
    train_slicer=SliceByTimeEventsTargets(slicing_time_window, overlap=slicing_time_window-train_stride_time, \
                    seq_length=args.train_length, seq_stride=args.train_stride, include_incomplete=True)
    val_slicer=SliceByTimeEventsTargets(slicing_time_window, overlap=slicing_time_window-valid_stride_time, \
                    seq_length=args.val_length, seq_stride=args.val_stride, include_incomplete=True)
    
    post_slicer_transform = transforms.Compose([
        SliceLongEventsToShort(time_window=int(10000/temp_subsample_factor), overlap=0, include_incomplete=True),
        EventSlicesToMap(sensor_size=(int(640*factor), int(480*factor), 2), \
                                n_time_bins=args.n_time_bins, per_channel_normalize=args.voxel_grid_ch_normalization,
                                map_type=args.map_type)
    ])
    # train_data = SlicedDataset(train_data_orig, train_slicer, transform=post_slicer_transform)
    # val_data = SlicedDataset(val_data_orig, val_slicer, transform=post_slicer_transform)
    # pdb.set_trace()
    if args.dataset == "t":
        train_data = SlicedDataset(train_data_orig, train_slicer, transform=post_slicer_transform, metadata_path=f"{args.metadata_dir}/3et_train_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}_t{args.map_type}")
        val_data = SlicedDataset(val_data_orig, val_slicer, transform=post_slicer_transform, metadata_path=f"{args.metadata_dir}/3et_val_vl_{args.val_length}_vs{args.val_stride}_ch{args.n_time_bins}_t{args.map_type}")
        #train_data = DiskCachedDataset(train_data, 
                           #cache_path=f"{args.cache_dir}/train_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}_t{args.map_type}",
                           #transforms=Jitter())
        #val_data = DiskCachedDataset(val_data, cache_path=f"{args.cache_dir}/val_vl_{args.val_length}_vs{args.val_stride}_ch{args.n_time_bins}_t{args.map_type}",
                           #transforms=None)
    else:
        train_data = SlicedDataset(train_data_orig, train_slicer, transform=post_slicer_transform, metadata_path=f"{args.metadata_dir}/3et_train_tl_{args.train_length}_ts{args.train_stride}_{args.dataset}")
        val_data = SlicedDataset(val_data_orig, val_slicer, transform=post_slicer_transform, metadata_path=f"{args.metadata_dir}/3et_val_vl_{args.val_length}_vs{args.val_stride}_{args.dataset}")
        train_data = DiskCachedDataset(train_data, 
                                   cache_path=f"{args.cache_dir}/train_tl_{args.train_length}_ts{args.train_stride}_{args.dataset}",
                                   transforms=Jitter())
        val_data = DiskCachedDataset(val_data, cache_path=f"{args.cache_dir}/val_vl_{args.val_length}_vs{args.val_stride}_{args.dataset}",
                                   transforms=None)
    device = "cuda"
    # Finally we wrap the dataset with pytorch dataloader
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, generator=torch.Generator(device=device), shuffle=True, \
                                pin_memory=False)
    
    val_loader = DataLoader(val_data, batch_size=args.batch_size, generator=torch.Generator(device=device), shuffle=False \
                            )
    
    
    return train_loader, val_loader
