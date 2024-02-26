import soundfile as sf
import numpy as np
import librosa
from model import RNNModel, SpectrogramDataset, preprocess_data
import torch 
from datetime import datetime

n_fft=2048
hop_length=512
win_length=2048
sequence_length = 20
file_name = "../assets/Wiley.wav"
x_frames, y_frames = preprocess_data(file_name, n_fft=n_fft, 
                                     hop_length=hop_length, win_length=win_length, 
                                     sequence_length=sequence_length)
spectrogram_dataset = SpectrogramDataset(x_frames, y_frames)


points = [0.0, 0.5, 0.2, 0.7]
lengths = [200, 200, 200, 200] 
random_strength = 0.2

model = RNNModel(input_size=1025, hidden_size=128, num_layers=2, output_size=1025)  # Example model initialization
checkpoint = "model_weights_26-Feb-2024-17-10-36.pth"
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
predicted_magnitudes = impulse
random_chance = 0.05
print(x_frames.shape, impulse.shape)

#x_frames is items x fft size x seq_len
#impulse is fft size x seq_len
#predicted_magnitudes is num_frames x fft size
#model takes 1 x fft size x seq_len
#model gives 1 x fft size
#the confusing this is the impulse has sequence after fft, and the predicted mags has it before 
#prediction.transpose(0,1) when joining to fix this

for j in range(output_sequence_length):
    prediction = model(impulse.unsqueeze(0))
    predicted_magnitudes = torch.cat((predicted_magnitudes, prediction.transpose(0,1)), dim=1)
    impulse = predicted_magnitudes[:,-sequence_length:]
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

# y_frames = y_frames.detach().numpy()
# print(y_frames.shape)

# from scipy.signal import get_window
# reconstructed_signal = np.zeros(len(y_frames * hop_length) + win_length)
# start = 0
# window = get_window('hann', win_length)
# for f in y_frames:
#     print(f.shape)
#     chunk = librosa.griffinlim(f.reshape(1, -1), n_fft=2028)
#     windowed_chunk = chunk * window
#     reconstructed_signal[start:start+win_length] += windowed_chunk
#     start += hop_length
# timestampStr = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
# # # WRITE AUDIO
# output_name = "gospel"
# sf.write(f"{output_name}_{timestampStr}.wav", reconstructed_signal, 22050)

predicted_magnitudes = predicted_magnitudes.detach().numpy()
audio = librosa.griffinlim(predicted_magnitudes, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
print(predicted_magnitudes.shape, len(audio))
timestampStr = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
# # WRITE AUDIO
output_name = "wiley"
sf.write(f"{output_name}_{timestampStr}.wav", audio, 44100)