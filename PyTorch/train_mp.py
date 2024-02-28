import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import  RNNModel
from model import preprocess_data_mp, MatchingPursuitDataset, get_dictionary
from datetime import datetime

sequence_length = 40
num_atoms = 100
dictionary_size = 10000
file_name = "../assets/Wiley_10.wav"
chunk_size = 2048
hop_length = chunk_size//4
sr = 44100
dictionary = get_dictionary(chunk_size=chunk_size, max_freq=10000, sr=sr)
x_frames, y_frames, cmin, cmax = preprocess_data_mp(file_name,sequence_length=sequence_length, 
                                        sr = sr, num_atoms=num_atoms,
                                        chunk_size=chunk_size, hop_length=hop_length, 
                                        dictionary=dictionary)
dataset = MatchingPursuitDataset(x_frames, y_frames)
# Create a DataLoader
batch_size = 64  # Define your batch size
shuffle = True   # Shuffle the data every epoch

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
# # Model parameters
learning_rate = 0.01
amount_epochs = 200
batch_size = 64
weight_decay = 0.0001

# loss_function = MultiClassCoeffiecentLoss(num_atoms, dictionary_size)
# model = RNNEmbeddingModel(num_categories=dictionary_size, num_atoms=num_atoms, embedding_dim=50,
#                           hidden_size=128, num_layers=2) 

# for epoch in range(amount_epochs):
#     running_loss = 0.0
#     for inputs, targets in dataloader:
#         opt.zero_grad()
#         predicted_indices, predicted_coefficients = model(inputs)
#         predicted_indices = torch.argmax(predicted_indices, dim=2)
#         loss = loss_function(predicted_indices, predicted_coefficients, targets)
#         loss.backward()
#         opt.step()
#         running_loss += loss.item()
#     print(f'Epoch [{epoch+1}/{amount_epochs}], Loss: {running_loss/len(dataloader):.4f}')
#     running_loss = 0.0

loss_type = nn.MSELoss()
model = RNNModel(input_size=num_atoms*2, hidden_size=128, num_layers=2, output_size=num_atoms*2) 

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

