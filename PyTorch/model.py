# Recurrent Neural Network
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np
import librosa
import sys
from os.path import isdir
from os import listdir


def preprocess_data(path, n_fft=2048,hop_length=512, win_length=2048, sequence_length = 40, sr = 44100):
    x = [0]
    if not isdir(path):
        x, sr = librosa.load(path, sr=sr) 
    else:
        files = listdir(path)
        x = np.array([0])
        for file in files:
            if not ".DS" in file:
                audio, sr, = librosa.load(path + file, sr = sr)
                x = np.concatenate((x, audio))

    x = np.array(x, dtype=np.float32) 
    data_tf = torch.tensor(x)
    # Compute STFT
    n = torch.stft(data_tf, n_fft=n_fft, hop_length=hop_length, win_length=win_length, 
                window=torch.hann_window(win_length), center=True, normalized=False, onesided=True, return_complex=True)

    magnitude_spectrograms = torch.abs(n)
    print(data_tf.shape, n.shape, magnitude_spectrograms.shape)

    start = 0
    end = magnitude_spectrograms.shape[1] - sequence_length - 1 
    step = 1
    x_frames = []
    y_frames = []
    
    for i in range(start, end, step):
        done = int((float(i) / float(end)) * 100.0)
        sys.stdout.write('{}% data generation complete.   \r'.format(done))
        sys.stdout.flush()
        x = magnitude_spectrograms[:, i:i + sequence_length]
        y = magnitude_spectrograms[:, i + sequence_length]
        x_frames.append(x)
        y_frames.append(y)

    x_frames = torch.stack(x_frames)
    y_frames = torch.stack(y_frames)
    print(x_frames.shape, y_frames.shape)
    return x_frames, y_frames

class SpectrogramDataset(Dataset):
    def __init__(self, x_frames, y_frames):
        self.x_frames = x_frames
        self.y_frames = y_frames

    def __len__(self):
        return self.x_frames.shape[0]  # Number of frames

    def __getitem__(self, idx):
        return self.x_frames[idx], self.y_frames[idx]

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
    
        self.batch_norm = nn.BatchNorm1d(input_size)
        print(input_size, hidden_size, num_layers)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.batch_norm(x) # BatchNorm expects [batch, features, seq_len]
        x, _ = self.lstm(x.transpose(1, 2))  # lstm expects [batch, seq_len, features]
        x = self.fc(x[:, -1, :]) 
        return x
    
