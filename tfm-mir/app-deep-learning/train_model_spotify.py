from common import GENRES
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout, Activation, LSTM, \
        TimeDistributed, Convolution1D, MaxPooling1D,Conv1D,AveragePooling1D, Flatten,GlobalAveragePooling1D,GlobalMaxPooling1D,concatenate
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
from optparse import OptionParser
import os


SEED = 42
N_LAYERS = 3
FILTER_LENGTH = 5
CONV_FILTER_COUNT = 256
LSTM_COUNT = 256
BATCH_SIZE = 32
EPOCH_COUNT = 60


def train_model(data, model_path):
    x = data['x']
    y = data['y']
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    print ("Scaling data ...")
    for val in range(x.shape[0]):
        x[val] = scaler.fit_transform(x[val])
        if val%500 == 0 :
            print(val)
               
    (x_train, x_val, y_train, y_val) = train_test_split(x, y, test_size=0.2,
            random_state=SEED)

    print('Building model...')

    input_shape = (x_train.shape[1], x_train.shape[2])

    model_input = Input(shape=input_shape)
    layer = model_input
    for i in range(3):
        layer = Conv1D(filters=256, kernel_size=4,strides=2)(layer)
        layer = Activation('relu')(layer)
        layer = MaxPooling1D(2)(layer)
    averagePool = GlobalAveragePooling1D()(layer)
    maxPool = GlobalMaxPooling1D()(layer)
    layer = concatenate([averagePool, maxPool])
    layer = Dropout(rate=0.5)(layer)
    layer = Dense(units=len(GENRES))(layer)
    model_output = Activation('softmax')(layer)
    model = Model(model_input, model_output)
    opt = Adam()
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    model.fit(x_train, y_train,batch_size=BATCH_SIZE,epochs=80,validation_data=(x_val, y_val),verbose=1)
    return model
    
    model.summary()

    print('Training...')
    
    earlyStop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
    model.fit(
        x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH_COUNT,
        validation_data=(x_val, y_val), verbose=1, callbacks=[
            ModelCheckpoint(
                model_path, save_best_only=True, monitor='val_acc', verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_acc', factor=0.5, patience=10, min_delta=0.01,
                verbose=1
            ),
            earlyStop
        ]
    )

    return model

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-d', '--data_path', dest='data_path',
            default=os.path.join(os.path.dirname(__file__),
                'data/data.pkl'),
            help='path to the data pickle', metavar='DATA_PATH')
    parser.add_option('-m', '--model_path', dest='model_path',
            default=os.path.join(os.path.dirname(__file__),
                'models/model-spotify.h5'),
            help='path to the output model HDF5 file', metavar='MODEL_PATH')
    options, args = parser.parse_args()

    with open(options.data_path, 'rb') as f:
        data = pickle.load(f)

    train_model(data, options.model_path)
