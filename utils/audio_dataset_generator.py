import os
import sys
import random
import librosa
import pywt
import math
import numpy as np
import tensorflow as tf


class AudioDatasetGenerator:
    """
    Class to manage the dataset for audio generation.
    """

    def __init__(self, fft_size=2048, window_size=1024, hop_size=512,
                 sequence_length=16, sample_rate=44100):
        """Inits the class. Set the fft values to have a significant effect on
        the training of the neural network."""
        self.counter = 0
        self.epoch_count = 0
        self.previous_epoch = -1
        self.x_frames = []
        self.y_frames = []
        self.fft_size = fft_size
        self.window_size = window_size
        self.hop_size = hop_size
        self.sequence_length = sequence_length
        self.sample_rate = sample_rate

    def load(self, data_path, force=False):
        """Loads the dataset from either the binary numpy file, or generates
        from a folder of wav files specified at the data_path."""
        x_frames_name = os.path.join(data_path, "x_frames.npy")
        y_frames_name = os.path.join(data_path, "y_frames.npy")
        if os.path.isfile(x_frames_name) and os.path.isfile(y_frames_name) and force == False:
            self.x_frames = np.load(x_frames_name)
            self.y_frames = np.load(y_frames_name)
        elif os.path.exists(data_path):
            self._generate_data(data_path)
            self.x_frames = np.array(self.x_frames)
            self.y_frames = np.array(self.y_frames)
            self.x_frames, self.y_frames = self.unison_shuffled_copies(self.x_frames,
                                                                       self.y_frames)
            np.save(x_frames_name, self.x_frames)
            np.save(y_frames_name, self.y_frames)
        else:
            raise ValueError("Couldn't load files from the supplied path.")

    def get_next_batch(self, batch_size):
        """Gets a new batch. Reshuffles the dataset at the end of the epoch."""
        if self.counter + batch_size > len(self.y_frames):
            self.counter = 0
            self.epoch_count += 1
            self.x_frames, self.y_frames = self.unison_shuffled_copies(self.x_frames,
                                                                       self.y_frames)
        return_x = self.x_frames[self.counter:self.counter + batch_size]
        return_y = self.y_frames[self.counter:self.counter + batch_size]
        self.counter += batch_size
        return return_x, return_y

    def is_new_epoch(self):
        """Returns true if there has been a new epoch."""
        if self.epoch_count != self.previous_epoch:
            self.previous_epoch = self.epoch_count
            return True
        return False

    def get_epoch(self):
        """Returns the current epoch."""
        return self.epoch_count

    def reset_epoch(self):
        """Resets the current epoch."""
        self.epoch_count = 0
        self.previous_epoch -1

    def get_x_shape(self):
        """Gets the shame for the x frames. Useful for placeholders."""
        return [None, self.x_frames.shape[1], self.x_frames.shape[2]]

    def get_y_shape(self):
        """Gets the shame for the y frames. Useful for placeholders."""
        return [None, self.y_frames.shape[1]]

    def completed_all_epochs(self, desired_epochs):
        """Returns true once the get next batch method has been called enough
        to have run through desired_epochs amount of epochs."""
        return self.epoch_count >= desired_epochs

    def unison_shuffled_copies(self, a, b):
        """Shuffle NumPy arrays in unison."""
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def griffin_lim(self, stftm_matrix, max_iter=100):
        """"Iterative method to 'build' phases for magnitudes."""
        stft_matrix = np.random.random(stftm_matrix.shape)
        y = librosa.core.istft(stft_matrix, self.hop_size, self.window_size)
        for i in range(max_iter):
            stft_matrix = librosa.core.stft(y, self.fft_size, self.hop_size, self.window_size)
            stft_matrix = stftm_matrix * stft_matrix / np.abs(stft_matrix)
            y = librosa.core.istft(stft_matrix, self.hop_size, self.window_size)
        return y
    
    def generate_samples(self, prediction_tensor, x, training, keep_prob,
                         amount_samples=5, sequence_max_length=2000,
                         impulse_scale=666, griffin_iterations=100):
        """Generates samples in the supplied folder path."""
        all_audio = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(amount_samples):
                random_index = random.randint(0, (len(self.x_frames) - 1))

                impulse_shape = np.array(self.x_frames[random_index]).shape
                #impulse = np.random.random_sample(size=impulse_shape) * impulse_scale
                impulse = self.x_frames[random_index]
                predicted_magnitudes = impulse
                for j in range(sequence_max_length):
                    impulse = np.array(impulse).reshape(1,self.x_frames.shape[1], self.x_frames.shape[2])
                    
                    prediction = sess.run(prediction_tensor,
                    feed_dict={x: impulse, training: False, keep_prob: 1.0})
                    prediction = prediction.reshape(1, prediction.shape[1])
                    predicted_magnitudes = np.vstack((predicted_magnitudes, prediction))
                    impulse = predicted_magnitudes[-self.sequence_length:]

                predicted_magnitudes = np.array(predicted_magnitudes)
                all_audio += [self.griffin_lim(predicted_magnitudes.T, griffin_iterations)]
            return np.array(all_audio)

    def _generate_data(self, data_path):
        """Create some data from a folder of wav files.
        NOTE: the augmentation process should be parameterised."""
        file_names = os.listdir(data_path)
        fft_frames = []
        self.x_frames = []
        self.y_frames = []
        for file in file_names:
            if file.endswith('.wav'):
                file = os.path.join(data_path, file)
                data, sample_rate = librosa.load(file, sr=self.sample_rate,
                                                 mono=True)
                data = np.append(np.zeros(self.window_size * self.sequence_length), data)
                mags_phases = librosa.stft(data, n_fft=self.fft_size,
                                           win_length=self.window_size,
                                           hop_length=self.hop_size)
                magnitudes, phases = librosa.magphase(mags_phases)
                for magnitude_bins in magnitudes.T:
                    fft_frames += [magnitude_bins]  

        start = 0
        end = len(fft_frames) - self.sequence_length - 1
        step = 1
        for i in range(start, end, step):
            done = int(float(i) / float(end) * 100.0)
            sys.stdout.write('{}% data generation complete.   \r'.format(done))
            sys.stdout.flush()

            x = fft_frames[i:i + self.sequence_length]
            y = fft_frames[i + self.sequence_length]
            self.x_frames.append(x)
            self.y_frames.append(y)

        sys.stdout.write('100% data generation complete.')
        sys.stdout.flush()


class AudioWaveletDatasetGenerator:
    """
    Class for wavelets
    """

    def __init__(self, window_size=1024, sequence_length=16, sample_rate=44100, wavelet='db10'):
        """Inits the class. Set the fft values to have a significant effect on
        the training of the neural network."""
        self.x_frames = []
        self.y_frames = []
        self.window_size = window_size
        print(self.window_size)
        self.sample_rate = sample_rate
        self.sequence_length = sequence_length
        self.coeff_slices = []
        self.wavelet = wavelet
        
    def load(self, data_path, force=False):
        """Loads the dataset from either the binary numpy file, or generates
        from a folder of wav files specified at the data_path."""
        x_frames_name = os.path.join(data_path, "x_frames.npy")
        y_frames_name = os.path.join(data_path, "y_frames.npy")
        if os.path.isfile(x_frames_name) and os.path.isfile(y_frames_name) and force == False:
            self.x_frames = np.load(x_frames_name)
            self.y_frames = np.load(y_frames_name)
        elif os.path.exists(data_path):
            self._generate_data(data_path)
            self.x_frames = np.array(self.x_frames)
            self.y_frames = np.array(self.y_frames)
            self.x_frames, self.y_frames = self.unison_shuffled_copies(self.x_frames,
                                                                       self.y_frames)
            np.save(x_frames_name, self.x_frames)
            np.save(y_frames_name, self.y_frames)
        else:
            raise ValueError("Couldn't load files from the supplied path.")

    def unison_shuffled_copies(self, a, b):
        """Shuffle NumPy arrays in unison."""
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
            
    def _generate_data(self, data_path):
        """Create some data from a folder of wav files.
        NOTE: the augmentation process should be parameterised."""
        file_names = os.listdir(data_path)

        self.x_frames = []
        self.y_frames = []
        ws = self.window_size
        for file in file_names:
            if file.endswith('.wav'):
                file = os.path.join(data_path, file)
                data, sample_rate = librosa.load(file, sr=self.sample_rate, mono=True)
                data = np.append(np.zeros(self.window_size * self.sequence_length), data)
                for offset in range(0, 1):
                    wavelet_frames = []
                    sys.stdout.write('{} offset   \r'.format(offset))
                    sys.stdout.flush()
                    for i in range (0, math.floor((len(data)-offset)/float(ws))):
                        coeffs = pywt.wavedec(data[(i*ws)+offset:(i*ws+ws)+offset], self.wavelet)
                        coeff_arr, self.coeff_slices = pywt.coeffs_to_array(coeffs) #slices to flat array
                        wavelet_frames.append(coeff_arr) 

                    start = 0
                    end = len(wavelet_frames) - self.sequence_length
                    assert end > 0
                    step = 1
                    for i in range(start, end, step):
                        x = wavelet_frames[i:i + self.sequence_length]
                        y = wavelet_frames[i + self.sequence_length]
                        self.x_frames.append(x)
                        self.y_frames.append(y)
                done = int(float(offset) / float(1020) * 100.0)
                sys.stdout.write('{}% data generation complete.   \r'.format(done))
                sys.stdout.flush()
        sys.stdout.write('100% data generation complete.')
        sys.stdout.flush()
        