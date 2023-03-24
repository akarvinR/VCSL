import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from vcsl import VideoFramesDataset
from torch.utils.data import DataLoader
import requests
import argparse
import torch
from PIL import Image

from isc_feature_extractor import create_model

# the frames_all.csv must contain columns described in Frame Sampling
parser = argparse.ArgumentParser()
parser.add_argument("-outputfile", "--outputfile", help = "output file name")
parser.add_argument("-csvfile", "--csvfile", help = "csv file")
parser.add_argument("-acc", "--acc", help = "cpu or gpu")
parser.add_argument("-batchsize", "--batchsize", help = "")
parser.add_argument("-numwork", "--numwork", help = "")
args = parser.parse_args() 
"""
 python create_desc.py --outputfile query_desc.npz --csvfile ../data/test/query/query.csv --acc cuda --batchsize 5 --numwork 1
"""
df = pd.read_csv(args.csvfile)

data_list = df[['uuid', 'path', 'frame_count']].values.tolist()

data_transforms = [
    lambda x: preprocessor(x).numpy()
]

dataset = VideoFramesDataset(data_list,
                             id_to_key_fn=VideoFramesDataset.build_image_key,
                             transforms=data_transforms, path = None,
                             store_type="local")

loader = DataLoader(dataset, collate_fn=lambda x: x,
                    batch_size=int(args.batchsize),
                    num_workers=int(args.numwork))


recommended_weight_name = 'isc_ft_v107'
model, preprocessor = create_model(weight_name=recommended_weight_name, device=args.acc)

i = 0
query_desc = []
qry_timestamps = []
qry_video_ids = []

for batch_data in loader:
    # batch data: List[Tuple[str, int, np.ndarray]]
    video_ids, frame_ids, images = zip(*batch_data)

    print(i)
    i += len(video_ids)
    images = torch.Tensor(list(images)) 
  
    print(images.shape)

    y = model(images)

    query_desc += y.tolist()
    qry_video_ids += video_ids
    break

np.savez(
        args.outputfile,
        video_ids=qry_video_ids,
        features=query_desc,
        timestamps=qry_timestamps,
    )
