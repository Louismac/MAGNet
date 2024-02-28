# Recurrent Neural Network
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np
import librosa
import sys
from os.path import isdir, exists
from os import listdir
from matching_pursuit import create_gabor_dictionary, process_in_chunks

def get_dictionary(chunk_size=2048, dictionary_size = 10000, min_freq=30, max_freq=20000):
    sigmas = [0.05, 0.1, 0.2, 0.5, 0.7, 1.0, 1.5]  
    freqs = np.linspace(min_freq, max_freq, dictionary_size//len(sigmas))
    dictionary = create_gabor_dictionary(chunk_size, freqs, sigmas)
    dictionary = dictionary.astype(np.float64)
    dictionary /= np.linalg.norm(dictionary, axis=0)
    return dictionary

def normalise_array(arr):
    # Subtract the minimum and divide by the range
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalised_arr = (arr - min_val) / (max_val - min_val)
    return normalised_arr

def preprocess_data_mp(path, sr = 44100, chunk_size=2048, hop_length=1024, 
                       num_atoms=100, dictionary_size = 10000, sequence_length = 40):
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
    dictionary = get_dictionary(chunk_size=chunk_size)
    _,chunks_info = process_in_chunks(x, 
                                    dictionary, 
                                    hop_length=hop_length,
                                    chunk_size=chunk_size, 
                                    iterations=num_atoms)
    #chunks info is num_frames x (num_atoms * 2)
    start = 0
    end = len(chunks_info) - sequence_length - 1 
    step = 1
    x_frames = []
    y_frames = []
    print(np.unique(chunks_info[:,:num_atoms]))
    chunks_info[:,:num_atoms] = chunks_info[:,:num_atoms]/dictionary_size
    for i in range(start, end, step):
        x = chunks_info[i:i + sequence_length]
        y = chunks_info[i + sequence_length]
        x_frames.append(torch.tensor(x))
        y_frames.append(torch.tensor(y))
    x_frames = torch.stack(x_frames).transpose(1,2)
    y_frames = torch.stack(y_frames)
    print(x_frames.shape, y_frames.shape)
    return x_frames, y_frames

class MatchingPursuitDataset(Dataset):
    def __init__(self, x_frames, y_frames):
        self.x_frames = x_frames.float()
        self.y_frames = y_frames.float()

    def __len__(self):
        return self.x_frames.shape[0]  # Number of frames

    def __getitem__(self, idx):
        return self.x_frames[idx], self.y_frames[idx]
    
class MultiClassCoeffiecentLoss(nn.Module):
    def __init__(self, num_atoms, num_categories):
        super(MultiClassCoeffiecentLoss, self).__init__()
        self.num_atoms = num_atoms
        self.num_categories = num_categories
        self.softmax_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted_indices, predicted_coefficients, targets):
        true_indices = targets[:,:self.num_atoms]
        true_coefficients = targets[:,self.num_atoms:]
        indices_loss = self.softmax_loss(predicted_indices.float(), true_indices.float())
        coefficients_loss = self.mse_loss(predicted_coefficients, true_coefficients).float()
        total_loss = indices_loss + coefficients_loss
        return total_loss

class RNNEmbeddingModel(nn.Module):
    def __init__(self, num_categories, num_atoms,embedding_dim, hidden_size, num_layers):
        super(RNNEmbeddingModel, self).__init__()
        self.num_categories = num_categories
        self.num_atoms = num_atoms
        self.batch_norm = nn.BatchNorm1d(embedding_dim+1)
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        self.lstm = nn.LSTM((embedding_dim+1)*num_atoms, hidden_size, num_layers, batch_first=True)
        #this is num_categories for the indices and num_atoms for the coeffients
        self.fc = nn.Linear(hidden_size, (num_categories * num_atoms)+num_atoms) 
    
    def forward(self, x):
        indices = x[:, :self.num_atoms].long()
        coefficients = x[:, self.num_atoms:].float()
        embedded_indices = self.embedding(indices)
        concatenated_inputs = torch.cat((embedded_indices, coefficients.unsqueeze(-1)), dim=-1)
        reshaped_inputs = concatenated_inputs.view(-1, concatenated_inputs.size(-1))
        x = self.batch_norm(reshaped_inputs)
        x = x.view(concatenated_inputs.size())
        x = x.view(x.size(0), x.size(2), -1)
        # LSTM expects [batch, seq_len, features]
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        output_indices, output_coefficients = torch.split(x, [self.num_categories*self.num_atoms, self.num_atoms], dim=1)
        softmax_outputs = output_indices.view(-1, self.num_atoms, self.num_categories)
        return softmax_outputs.float(), output_coefficients

def preprocess_data(path, n_fft=2048,hop_length=512, win_length=2048, sequence_length = 40, sr = 44100):
    cached_x_path = path + '_x_frames.npy'
    cached_y_path = path + '_y_frames.npy'
    if exists(cached_x_path) and exists(cached_y_path):
        x_frames = np.load(cached_x_path)
        y_frames = np.load(cached_y_path)
        print("loading cached data")
        return torch.tensor(x_frames), torch.tensor(y_frames)
    
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
    np.save(cached_x_path, x_frames)
    np.save(cached_y_path, y_frames)
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
        x = x.float()
        x = self.batch_norm(x) # BatchNorm expects [batch, features, seq_len]
        x, _ = self.lstm(x.transpose(1, 2))  # lstm expects [batch, seq_len, features]
        x = self.fc(x[:, -1, :]) 
        return x
    
