from model import MatchingPursuitDataset
from model import preprocess_data_sparse, preprocess_data_normalised
from matching_pursuit import reconstruct_from_sparse_chunks, reconstruct_from_normalised_chunks, get_dictionary
from datetime import datetime
import soundfile as sf


sequence_length = 40
num_atoms = 200
dictionary_size = 20000
file_name = "../assets/Wiley_10.wav"
chunk_size = 2048
hop_length = chunk_size//4
sr = 44100
dictionary = get_dictionary(chunk_size=chunk_size, max_freq=10000, sr=sr)
x_frames, y_frames = preprocess_data_sparse(file_name,sequence_length=sequence_length, 
                                        sr = sr, num_atoms=num_atoms,
                                        chunk_size=chunk_size, hop_length=hop_length, 
                                        dictionary=dictionary)

# x_frames, y_frames, cmax, cmin = preprocess_data_normalised(file_name,sequence_length=sequence_length, 
#                                         sr = sr, num_atoms=num_atoms,
#                                         chunk_size=chunk_size, hop_length=hop_length, 
#                                         dictionary=dictionary)

dataset = MatchingPursuitDataset(x_frames, y_frames)

audio = reconstruct_from_sparse_chunks(y_frames, dictionary, chunk_size, hop_length)
# audio = reconstruct_from_normalised_chunks(y_frames, dictionary, chunk_size, hop_length, cmax, cmin)

print(y_frames.shape, len(audio))
timestampStr = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")

# # WRITE AUDIO
output_name = "wiley"
sf.write(f"{output_name}_{timestampStr}.wav", audio, 44100)
