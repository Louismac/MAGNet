import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import  RNNModel
from model import preprocess_data_sparse, MatchingPursuitDataset
from matching_pursuit import get_dictionary
from datetime import datetime

sequence_length = 40
num_atoms = 100
dictionary_size = 10000
file_name = "../assets/Wiley_10.wav"
chunk_size = 2048
hop_length = chunk_size//4
sr = 44100
dictionary = get_dictionary(chunk_size=chunk_size, max_freq=10000, sr=sr)
dictionary_size = len(dictionary[0])
x_frames, y_frames = preprocess_data_sparse(file_name,sequence_length=sequence_length, 
                                        sr = sr, num_atoms=num_atoms,
                                        chunk_size=chunk_size, hop_length=hop_length, 
                                        dictionary=dictionary)
dataset = MatchingPursuitDataset(x_frames, y_frames)
# Create a DataLoader
batch_size = 64  # Define your batch size
shuffle = True   # Shuffle the data every epoch

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
# # Model parameters
learning_rate = 0.001
amount_epochs = 10
weight_decay = 0.0001

loss_type = nn.MSELoss()
model = RNNModel(input_size=dictionary_size, hidden_size=256, num_layers=2, output_size=dictionary_size) 
opt = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# checkpoint = 'model_weights_26-Feb-2024-16-58-29.pth'
# model.load_state_dict(torch.load(checkpoint))
# model.eval() 

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

