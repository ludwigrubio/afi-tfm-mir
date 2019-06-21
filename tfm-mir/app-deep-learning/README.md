# CNN for Live Music Genre Recognition

Convolutional Neural Networks for Live Music Genre Recognition is an app re-implemented based on [Deep Sound](https://github.com/deepsound-project/genre-recognition)

## Demo

You can see a demo at [https://music.datalud.com](https://music.datalud.com). You can upload a song using the big (and only) button and see the results for yourself. All mp3 files should work fine.


## Usage

It's easiest to run using Docker:

```shell
docker build -t genre-recognition . && docker run -d -p 8080:80 genre-recognition
```

The demo will be accessible at http://0.0.0.0:8080/.

By default, it will use a model pretrained based on small FMA data set.

You can also provide your own model, as long as it matches the input and output architecture of the provided model.

If you wish to train a model by yourself, download the [GTZAN dataset](http://opihi.cs.uvic.ca/sound/genres.tar.gz) (or provide FMA data) to the data/ directory, extract it and follow next steps:
 
* Run `create_data_pickle.py` or `create_data_pickle_fma.py` to preprocess the data.
* Run `train_model.py` to train the model.
* Afterwards you should run `model_to_tfjs.py` to convert the model to TensorFlow.js so it can be served.

```shell
cd data
wget http://opihi.cs.uvic.ca/sound/genres.tar.gz
tar zxvf genres.tar.gz
cd ..
pip install -r requirements.txt or load __environment_dl.yml__ file in Anaconda.
python create_data_pickle_fma.py
python train_model.py
python model_to_tfjs.py
```

You can "visualize" the filters learned by the convolutional layers using `extract_filters.py`. This script for every neuron extracts and concatenates several chunks resulting in its maximum activation from the tracks of the data set. By default, it will put the visualizations in the filters/ directory. It requires the GTZAN data set and its pickled version in the data/ directory. Run the commands above to obtain them. You can control the number of extracted chunks using the --count0 argument. Extracting higher numbers of chunks will be slower.
