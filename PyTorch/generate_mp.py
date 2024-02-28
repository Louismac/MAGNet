import torch
from model import RNNModel, MatchingPursuitDataset
from model import get_dictionary, preprocess_data_mp
from matching_pursuit import reconstruct_from_chunks
from datetime import datetime
import numpy as np
import soundfile as sf

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

points = [0.0, 0.5, 0.2, 0.7]
lengths = [500, 500, 500, 500] 
random_strength = 0.2

model = RNNModel(input_size=num_atoms*2, hidden_size=128, num_layers=2, output_size=num_atoms*2)  # Example model initialization
checkpoint ="model_weights_28-Feb-2024-19-44-00.pth"
model.load_state_dict(torch.load(checkpoint))
model.eval()

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
    prediction = model(impulse.unsqueeze(0))
    predicted_atoms = torch.cat((predicted_atoms, prediction.transpose(0,1)), dim=1)
    impulse = predicted_atoms[:,-sequence_length:]
    if (np.random.random_sample() < random_chance) :
        np.random.seed()
        random_index = np.random.randint(0, (len(x_frames) - 1))                                                                                                                    
        impulse = x_frames[random_index]
    if j > change_at:
      print(ctr, j, change_at, index)
      ctr = ctr + 1
      index = int(points[ctr] * len(x_frames))
      impulse = x_frames[index]
      change_at = change_at + lengths[ctr]
audio = reconstruct_from_chunks(predicted_atoms, dictionary,chunk_size, hop_length,
                                cmin = cmin, cmax=cmax)

print(predicted_atoms.shape, len(audio))
timestampStr = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")

# # WRITE AUDIO
output_name = "wiley"
sf.write(f"{output_name}_{timestampStr}.wav", audio, 44100)