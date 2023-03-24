import os 
import argparse
import pandas as pd
# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-outputfile", "--outputfile", help = "output file name")
parser.add_argument("-path", "--path", help = "Enter data path")
 
args = parser.parse_args()
dict_ = {'uuid' : [], 'path' : [], 'frame_count': []}
path = args.path
"""
 python generate_dataframe.py --outputfile ../data/test/query/query.csv --path ../data/test/query 
"""
for filename in os.listdir(path):

    if os.path.isdir(os.path.join(path,filename)):
        folder_path = os.path.join(path,filename)
        dict_['uuid'].append(filename)
        dict_['path'].append(folder_path)
        dict_['frame_count'].append(len(os.listdir(folder_path)))

df = pd.DataFrame(dict_)
df.to_csv(args.outputfile)