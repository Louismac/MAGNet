import sys
import tensorflow as tf
import tflearn
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell, GRUCell
from tflearn.layers.core import dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from audio_dataset_generator import AudioDatasetGenerator
import random
import numpy as np
import json


def conv_net(net, filters, kernels, non_linearity):  
    """
    A quick function to build a conv net. 
    At the end it reshapes the network to be 3d to work with recurrent units.
    """
    assert len(filters) == len(kernels)
    
    for i in range(len(filters)):
        net = conv_2d(net, filters[i], kernels[i], activation=non_linearity)
        net = max_pool_2d(net, 2)
        
    dim1 = net.get_shape().as_list()[1]
    dim2 = net.get_shape().as_list()[2]
    dim3 = net.get_shape().as_list()[3]
    return tf.reshape(net, [-1, dim1 * dim3, dim2])
   
                      
def recurrent_net(net, rec_type, rec_size, return_sequence):
    """
    A quick if else block to build a recurrent layer, based on the type specified
    by the user.
    """
    if rec_type == 'lstm':
        net = tflearn.layers.recurrent.lstm(net, rec_size, return_seq=return_sequence)
    elif rec_type == 'gru':
        net = tflearn.layers.recurrent.gru(net, rec_size, return_seq=return_sequence)
    elif rec_type == 'bi_lstm':
        net = bidirectional_rnn(net, 
                                BasicLSTMCell(rec_size), 
                                BasicLSTMCell(rec_size), 
                                return_seq=return_sequence)
    elif rec_type == 'bi_gru':
        net = bidirectional_rnn(net, 
                                GRUCell(rec_size), 
                                GRUCell(rec_size), 
                                return_seq=return_sequence)
    else:
        raise ValueError('Incorrect rnn type passed. Try lstm, gru, bi_lstm or bi_gru.')
    return net


def create_random_parameters():
    hyperparameters = dict()
    
    # Dataset
    hyperparameters['sequence_length']      = random.choice([40, 50, 60, 70, 80])

    # Feature Extraction and Audio Genreation
    hyperparameters['sample_rate']          = 22050
    hyperparameters['fft_size']             = 2048
    hyperparameters['window_size']          = 1024
    hyperparameters['hop_size']             = 512

    # General Network
    hyperparameters['learning_rate']        = random.choice([1e-2, 1e-3, 1e-4])
    hyperparameters['amount_epochs']        = 700
    hyperparameters['batch_size']           = random.choice([32, 64, 128, 256])
    hyperparameters['keep_prob']            = random.choice([0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    hyperparameters['activation']           = random.choice(['sigmoid', 'tanh', 'relu', 'leaky_relu', 'selu'])
    hyperparameters['optimiser']            = random.choice(['adam', 'rmsprop'])
    hyperparameters['fully_connected_dim']  = random.choice([512, 1024, 2048])

    # Recurrent Neural Network
    hyperparameters['rnn_type']             = random.choice(["lstm", "gru", "bi_lstm", "bi_gru"])
    hyperparameters['number_rnn_layers']    = random.choice([1, 2, 3, 4])
    hyperparameters['rnn_number_units']     = random.choice([256, 512, 1024])

    # Convolutional Neural Network
    hyperparameters['use_cnn']              = random.choice([True, False])
    cnn_int                                 = random.randint(0, 3)
    hyperparameters['number_filters']       = [[32], [64], [32, 64], [64, 32]][cnn_int]
    hyperparameters['filter_sizes']         = [[1], [3], [1, 5], [1, 3]][cnn_int]
    
    hyperparameters['fitness']              = 0.0
    
    return hyperparameters


epoch = 0

for model_no in range(100):
    try:
        hyperparameters = create_random_parameters()

        paths = ["assets/electronic_piano/", "assets/other", "assets/test_samples/"]

        for audio_data_path in paths:

            tf.reset_default_graph()

            dataset = AudioDatasetGenerator(hyperparameters['fft_size'], 
                                            hyperparameters['window_size'], 
                                            hyperparameters['hop_size'],
                                            hyperparameters['sequence_length'], 
                                            hyperparameters['sample_rate'])

            dataset.load(audio_data_path, True)

            if hyperparameters['use_cnn']:
                dataset.x_frames = dataset.x_frames.reshape(dataset.x_frames.shape[0], 
                                                            dataset.x_frames.shape[1], 
                                                            dataset.x_frames.shape[2], 
                                                            1)
            if hyperparameters['use_cnn']:
                net = tflearn.input_data([None, 
                                          dataset.x_frames.shape[1], 
                                          dataset.x_frames.shape[2], 
                                          dataset.x_frames.shape[3]], 
                                         name="input_data0")
                net = conv_net(net, 
                               hyperparameters['number_filters'], 
                               hyperparameters['filter_sizes'],
                               hyperparameters['activation'])
            else:                  
                net = tflearn.input_data([None, 
                                          dataset.x_frames.shape[1], 
                                          dataset.x_frames.shape[2]], 
                                         name="input_data0") 

            # Batch Norm
            net = tflearn.batch_normalization(net, name="batch_norm0")

            # Recurrent
            for layer in range(hyperparameters['number_rnn_layers']):
                return_sequence = not layer == (hyperparameters['number_rnn_layers'] - 1)
                net = recurrent_net(net, 
                                    hyperparameters['rnn_type'], 
                                    hyperparameters['rnn_number_units'], 
                                    return_sequence)
                if hyperparameters['keep_prob'] < 1.0:
                    net = dropout(net, 1.0 - hyperparameters['keep_prob'])

            # Dense + MLP Out
            net = tflearn.fully_connected(net, 
                                          dataset.y_frames.shape[1], 
                                          activation=hyperparameters['activation'],                                            
                                          regularizer='L2', 
                                          weight_decay=0.001)

            net = tflearn.fully_connected(net, 
                                          dataset.y_frames.shape[1], 
                                          activation='linear')

            net = tflearn.regression(net, 
                                     optimizer=hyperparameters['optimiser'],
                                     learning_rate=hyperparameters['learning_rate'],                                 
                                     loss="mean_square")

            model = tflearn.DNN(net, tensorboard_verbose=1)

            model.fit(dataset.x_frames, 
                      dataset.y_frames, 
                      show_metric=True, 
                      batch_size=hyperparameters['batch_size'], 
                      n_epoch=hyperparameters['amount_epochs'])

            model_name = '{}_{}'.format(epoch, model_no)
            with open(model_name + '.json', 'w') as fp:
                json.dump(hyperparameters, fp)

            amount_samples      = 1
            sequence_length_max = 1000
            impulse_scale       = 1.0
            griffin_iterations  = 60
            random_chance       = 0.0
            random_strength     = 0.0

            dimension1 = dataset.x_frames.shape[1]
            dimension2 = dataset.x_frames.shape[2]
            shape = (1, dimension1, dimension2, 1) if hyperparameters['use_cnn'] else (1, dimension1, dimension2)

            audio = []

            for i in range(amount_samples):                                                                                                                                   

                random_index = 5                                                                                                                    

                impulse = np.array(dataset.x_frames[random_index]) * impulse_scale
                predicted_magnitudes = impulse

                for j in range(sequence_length_max):

                    prediction = model.predict(impulse.reshape(shape))

                    if hyperparameters['use_cnn']:
                        prediction = prediction.reshape(1, dataset.y_frames.shape[1], 1)

                    predicted_magnitudes = np.vstack((predicted_magnitudes, prediction))                                                                                                  
                    impulse = predicted_magnitudes[-sequence_length:]

                    if (np.random.random_sample() < random_chance) :
                        idx = np.random.randint(0, dataset.sequence_length)
                        impulse[idx] = impulse[idx] + np.random.random_sample(impulse[idx].shape) * random_strength

                predicted_magnitudes = np.array(predicted_magnitudes).reshape(-1, window_size+1)                                                                           
                audio = np.array(dataset.griffin_lim(predicted_magnitudes.T, griffin_iterations))
                filepath = model_name + '_{}_{}.wav'.format(i, audio_data_path)
                librosa.output.write_wav(filepath, 
                                         audio, 
                                         hyperparameters['sample_rate'])
        
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as ex:
        print(ex)
        pass
