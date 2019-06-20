from common import load_track, GENRES
import numpy as np
from math import pi
import pickle
from pickle import dump
import os
import pandas as pd
from optparse import OptionParser

TRACK_COUNT = 8000

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
    
    dataset_path = '../data/fma_small'
    metadata_path  = '../data/fma_metadata/tracks.pkl'
    
    default_shape = get_default_shape(dataset_path)
    tracks = pickle.load(open(metadata_path, 'rb'))
    tracks = tracks[tracks['set', 'subset'] <= 'small']
    
    print(tracks.shape)

    x = np.zeros((TRACK_COUNT,) + default_shape, dtype=np.float32)
    y = np.zeros((TRACK_COUNT, len(GENRES)), dtype=np.float32)
    track_paths = {}
    
    '''
    for (genre_index, genre_name) in enumerate(GENRES):
        for i in range(TRACK_COUNT // len(GENRES)):
            file_name = '{}/{}.000{}.au'.format(genre_name,
                    genre_name, str(i).zfill(2))
            print('Processing', file_name)
            path = os.path.join(dataset_path, file_name)
            track_index = genre_index  * (TRACK_COUNT // len(GENRES)) + i
            x[track_index], _ = load_track(path, default_shape)
            y[track_index, genre_index] = 1
            track_paths[track_index] = os.path.abspath(path)
    '''
    return (x, y, track_paths)

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-d', '--dataset_path', dest='dataset_path',
            default=os.path.join(os.path.dirname(__file__), '../data/fma_small'),
            help='path to the GTZAN dataset directory', metavar='DATASET_PATH')
    parser.add_option('-m', '--metadata_path', dest='metadata_path',
            default=os.path.join(os.path.dirname(__file__),
                '../data/fma_metadata/tracks.pkl'),
            help='path to the metadata pkl', metavar='META_PATH')
    parser.add_option('-o', '--output_pkl_path', dest='output_pkl_path',
            default=os.path.join(os.path.dirname(__file__), 'data/data.pkl'),
            help='path to the output pickle', metavar='OUTPUT_PKL_PATH')
    options, args = parser.parse_args()

    (x, y, track_paths) = collect_data(options.dataset_path, options.metadata_path)

    data = {'x': x, 'y': y, 'track_paths': track_paths}
    with open(options.output_pkl_path, 'wb') as f:
        dump(data, f)
