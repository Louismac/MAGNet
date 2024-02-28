from model import MatchingPursuitDataset
from model import get_dictionary, preprocess_data_mp
from matching_pursuit import reconstruct_from_chunks
from datetime import datetime
import soundfile as sf
sequence_length = 40
num_atoms = 100

file_name = "../assets/Wiley.wav"
x_frames, y_frames = preprocess_data_mp(file_name, sr = 44100, num_atoms=num_atoms)
dataset = MatchingPursuitDataset(x_frames, y_frames)

chunk_size = 2048
dictionary = get_dictionary(chunk_size=chunk_size)
audio = reconstruct_from_chunks(y_frames.detach().numpy(), dictionary, chunk_size, chunk_size//2)

print(y_frames.shape, len(audio))
timestampStr = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")

# # WRITE AUDIO
output_name = "wiley"
sf.write(f"{output_name}_{timestampStr}.wav", audio, 44100)
