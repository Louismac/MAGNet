import torch
from model import MatchingPursuitDataset, RNNEmbeddingModel
from model import preprocess_data_embedding
from matching_pursuit import get_dictionary, reconstruct_from_embedding_chunks
from datetime import datetime
import numpy as np
import soundfile as sf
import sys

sequence_length = 40
num_atoms = 100
dictionary_size = 10000
file_name = "../assets/Wiley_10.wav"
chunk_size = 2048
hop_length = chunk_size//4
sr = 44100
dictionary = get_dictionary(chunk_size=chunk_size, max_freq=10000, sr=sr)
dictionary_size = len(dictionary[0])
x_frames, y_frames = preprocess_data_embedding(file_name,sequence_length=sequence_length, 
                                        sr = sr, num_atoms=num_atoms,
                                        chunk_size=chunk_size, hop_length=hop_length, 
                                        dictionary=dictionary)
dataset = MatchingPursuitDataset(x_frames, y_frames)

model = RNNEmbeddingModel(num_categories=dictionary_size, num_atoms=num_atoms, embedding_dim=300,
                          hidden_size=128, num_layers=2) 
checkpoint ="model_weights_29-Feb-2024-12-10-40.pth"
model.load_state_dict(torch.load(checkpoint))
model.eval()

points = [0.0]
lengths = [500] 
random_strength = 0.2

output_sequence_length = np.array(lengths).sum()
dimension1 = x_frames.shape[1]
dimension2 = x_frames.shape[2]
shape = (1, dimension1, dimension2)
ctr = 0
change_at = lengths[ctr]

audio = []
index = int(points[ctr] * len(x_frames))
impulse = x_frames[index]
predicted_atoms = impulse
random_chance = 0.05
print(x_frames.shape, impulse.shape)

for j in range(output_sequence_length):
    probs, coeffiecents = model(impulse.unsqueeze(0))
    values, indices = torch.topk(probs, num_atoms, dim=1)
    prediction = torch.cat((indices.squeeze(), coeffiecents.squeeze())).unsqueeze(0)
    predicted_atoms = torch.cat((predicted_atoms, prediction.transpose(0,1)), dim=1)
    impulse = predicted_atoms[:,-sequence_length:]
    if (np.random.random_sample() < random_chance) :
        np.random.seed()
        random_index = np.random.randint(0, (len(x_frames) - 1))                                                                                                                    
        impulse = x_frames[random_index]
    if j > change_at:
      ctr = ctr + 1
      index = int(points[ctr] * len(x_frames))
      impulse = x_frames[index]
      change_at = change_at + lengths[ctr]
    sys.stdout.write('{:.2f}% data generation complete.   \r'.format((j/output_sequence_length)*100))
    sys.stdout.flush()
audio = reconstruct_from_embedding_chunks(predicted_atoms.T, dictionary, chunk_size, hop_length)
print(predicted_atoms.T.shape, len(audio))
timestampStr = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")

# # WRITE AUDIO
output_name = "wiley"
sf.write(f"{output_name}_{timestampStr}.wav", audio, 44100)