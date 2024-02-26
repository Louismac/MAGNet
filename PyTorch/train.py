import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import RNNModel, SpectrogramDataset, preprocess_data
from datetime import datetime

n_fft = 2048
hop_length = 512
win_length = 2048
sequence_length = 40
file_name = "../assets/Wiley.wav"
x_frames, y_frames = preprocess_data(file_name, n_fft=n_fft, 
                                     hop_length=hop_length, win_length=win_length, 
                                     sequence_length=sequence_length, sr = 44100)
# Create an instance of the dataset
spectrogram_dataset = SpectrogramDataset(x_frames, y_frames)

# Create a DataLoader
batch_size = 64  # Define your batch size
shuffle = True   # Shuffle the data every epoch

dataloader = DataLoader(spectrogram_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

# # Model parameters
learning_rate = 0.001
amount_epochs = 200
batch_size = 64
loss_type = nn.MSELoss()
weight_decay = 0.0001

model = RNNModel(input_size=n_fft//2+1, hidden_size=128, num_layers=2, output_size=n_fft//2+1) 

# checkpoint = 'model_weights_26-Feb-2024-16-58-29.pth'
# model.load_state_dict(torch.load(checkpoint))
# model.eval() 

opt = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

for epoch in range(amount_epochs):
    running_loss = 0.0
    for inputs, targets in dataloader:
        opt.zero_grad()
        outputs = model(inputs)
        loss = loss_type(outputs, targets)
        loss.backward()
        opt.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{amount_epochs}], Loss: {running_loss/len(dataloader):.4f}')
    running_loss = 0.0

timestampStr = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
torch.save(model.state_dict(), f"model_weights_{timestampStr}.pth")
