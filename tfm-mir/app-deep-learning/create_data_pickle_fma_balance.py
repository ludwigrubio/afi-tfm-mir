from common import load_track, GENRES
from math import pi
from pickle import dump
from optparse import OptionParser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from imblearn.under_sampling import RandomUnderSampler


import pickle
import os
import pandas as pd
import numpy as np


def get_default_shape(dataset_path):
    tmp_features, _ = load_track(os.path.join(dataset_path,
        '000/000002.mp3'))
    return tmp_features.shape

def collect_data(dataset_path, metadata_path):
    '''
    Collects data from the FMA dataset into a pickle. Computes a Mel-scaled
    power spectrogram for each track.

    :param dataset_path: path to the FMA dataset directory
    :param dataset_metadata: path to the FMA metadata file
    :returns: triple (x, y, track_paths) where x is a matrix containing
        extracted features, y is a one-hot matrix of genre labels and
        track_paths is a dict of absolute track paths indexed by row indices in
        the x and y matrices
    '''   
    default_shape = get_default_shape(dataset_path)
    
    tracks = pickle.load(open(metadata_path, 'rb'))
    tracks = tracks[tracks['set', 'subset'] <= 'medium']
    
    empty_files = np.array(['001486', '005574', '065753', '080391', '098558', 
                            '098559', '098560', '098571', '099134', '105247',
                            '108925', '127336', '133297', '143992'])
    
    for x in empty_files:
       tracks = tracks.drop(int(x))
     
    tracks.reset_index(inplace= True)   
    t =  tracks['track_id']
    
    
    # Resample
    rus = RandomUnderSampler(random_state=1212)
    X_resampled, y_resampled = rus.fit_sample(np.array(t).reshape(-1, 1),tracks['track','genre_top'])

    X_resampled = X_resampled.reshape(336, )
    y_resampled = y_resampled.reshape(336, )
    TRACK_COUNT = X_resampled.shape[0]

    x = np.zeros((TRACK_COUNT,) + default_shape, dtype=np.float32)
    y = np.zeros((TRACK_COUNT, len(GENRES)), dtype=np.float32)
    track_paths = {}
    

    for i in range(TRACK_COUNT):
        tid_str = '{:06d}'.format(X_resampled[i])
      
        file_name = os.path.join(dataset_path, tid_str[:3], tid_str + '.mp3')
        print(f"Processing {file_name} - {i}")

        track_index = i 
        
        x[track_index], _ = load_track(file_name, default_shape)
        y[track_index, GENRES.index(y_resampled[i])] = 1
      
        track_paths[track_index] = os.path.abspath(file_name)
    

    return (x, y, track_paths)

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-d', '--dataset_path', dest='dataset_path',
            default=os.path.join(os.path.dirname(__file__), '../data/fma_medium/'),
            help='path to the GTZAN dataset directory', metavar='DATASET_PATH')
    parser.add_option('-m', '--metadata_path', dest='metadata_path',
            default=os.path.join(os.path.dirname(__file__),
                '../data/fma_metadata/tracks.pkl'),
            help='path to the metadata pkl', metavar='META_PATH')
    parser.add_option('-o', '--output_pkl_path', dest='output_pkl_path',
            default=os.path.join(os.path.dirname(__file__), 'data/data-balanced.pkl'),
            help='path to the output pickle', metavar='OUTPUT_PKL_PATH')
    options, args = parser.parse_args()

    (x, y, track_paths) = collect_data(options.dataset_path, options.metadata_path)

    data = {'x': x, 'y': y, 'track_paths': track_paths}
    with open(options.output_pkl_path, 'wb') as f:
        dump(data, f)
