from model import MatchingPursuitDataset
from model import get_dictionary, preprocess_data_mp
from matching_pursuit import reconstruct_from_chunks
from datetime import datetime
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

audio = reconstruct_from_chunks(y_frames, dictionary, chunk_size, hop_length,
                                cmin = cmin, cmax=cmax)

print(y_frames.shape, len(audio))
timestampStr = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")

# # WRITE AUDIO
output_name = "wiley"
sf.write(f"{output_name}_{timestampStr}.wav", audio, 44100)
