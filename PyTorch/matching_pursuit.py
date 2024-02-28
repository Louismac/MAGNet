import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
import numpy as np
from os.path import isdir, exists
from scipy.signal import get_window

def process_in_chunks(signal, dictionary, chunk_size=2048, hop_length = 1024,
                       window_type='hann', iterations = 100):

    cached_path = "cached_chunks_" + str(chunk_size) + "_" + str(len(dictionary[0])) + "_" + str(iterations) +".npy"
    reconstructed_signal = np.zeros(len(signal))
    if exists(cached_path):
        chunks_info = np.load(cached_path)
        print("loaded from cache")
        return reconstructed_signal, np.array(chunks_info)

    window = get_window(window_type, chunk_size)
    
    weight_sum = np.zeros(len(signal))
    chunks_info = []
    for start in range(0, len(signal) - chunk_size + 1, hop_length):
        end = start + chunk_size
        chunk = signal[start:end]
        windowed_chunk = chunk * window
        processed_chunk, atom_indices, coefficients = matching_pursuit(windowed_chunk, dictionary, iterations) 
        chunk_info = atom_indices + coefficients
        chunks_info.append(chunk_info)
        print("chunk complete", start, end, len(chunk_info), len(chunks_info))
        reconstructed_signal[start:end] += processed_chunk
        weight_sum[start:end] += window  
    # non_zero_weights = weight_sum > 0
    # reconstructed_signal[non_zero_weights] /= weight_sum[non_zero_weights]
    np.save(cached_path, chunks_info)
    return reconstructed_signal, np.array(chunks_info)

def matching_pursuit(signal, dictionary, iterations=20):

    residual = signal.copy() 
    reconstruction = np.zeros_like(signal)
    atom_indices = []
    coefficients = []

    for i in range(iterations):
        correlations = np.dot(dictionary.T, residual)  
        best_atom_index = np.argmax(np.abs(correlations))  
        best_coefficient = correlations[best_atom_index] 
        if not np.isinf(best_coefficient):
            reconstruction += best_coefficient * dictionary[:, best_atom_index] 
            residual = residual - (best_coefficient * dictionary[:, best_atom_index])  
            atom_indices.append(best_atom_index)
            coefficients.append(best_coefficient)
        else:
            break
    print(len(atom_indices))
    return reconstruction, atom_indices, coefficients

    # Instantiate the OMP model
    # omp = OrthogonalMatchingPursuit(n_nonzero_coefs=iterations)  # Adjust n_nonzero_coefs as needed

    # # Reshape signal for compatibility with OMP (OMP expects 2D input)
    # signal_reshaped = signal.reshape(-1, 1)
    # # Fit the model
    # omp.fit(dictionary, signal_reshaped)
    # # Get the coefficients (sparse representation)
    # coefficients = omp.coef_
    # reconstructed_signal = np.dot(dictionary, coefficients).flatten()
    # atom_indices = np.nonzero(coefficients)[0]
    # return reconstructed_signal, atom_indices, coefficients


def reconstruct_signal(atom_indices, coefficients, dictionary):
    reconstructed_signal = np.zeros(dictionary.shape[0])
    atom_indices = np.array(atom_indices, dtype=np.int32)
    for index, coeff in zip(atom_indices, coefficients):
        reconstructed_signal += coeff * dictionary[:, index]
    return reconstructed_signal

def reconstruct_from_chunks(chunks_info, dictionary, chunk_size=2048, hop_length=1024, cmin=0, cmax=1):
    
    signal_length = (len(chunks_info) * (hop_length))+chunk_size
    reconstructed_signal = np.zeros(signal_length)
    weight_sum = np.zeros(signal_length)  
    
    num_atoms = len(chunks_info[0])//2
    start = 0
    end = chunk_size
    
    for _, chunk_info in enumerate(chunks_info):

        atom_indices = chunk_info[:num_atoms].detach().numpy()
        atom_indices = np.array(np.ceil(atom_indices*len(dictionary[0])), dtype=np.int32)-1
        
        coefficients = chunk_info[num_atoms:].detach().numpy()
        coefficients = (coefficients * (cmax - cmin)) + cmin
        
        chunk_reconstruction = reconstruct_signal(atom_indices, coefficients, dictionary) 
        
        reconstructed_signal[start:end] += chunk_reconstruction
        weight_sum[start:end] += 1  
        start += hop_length
        end += hop_length

    overlap_areas = weight_sum > 1  
    reconstructed_signal[overlap_areas] /= weight_sum[overlap_areas]
    return reconstructed_signal

def generate_gabor_atom(length, freq, sigma, sr, phase=0):
    # Adjust time vector to be in seconds 
    t = np.linspace(-1, 1, length) * (length / sr)
    gaussian = np.exp(-0.5 * (t / sigma) ** 2)
    sinusoid = np.cos(2 * np.pi * freq * t + phase)
    return gaussian * sinusoid

def create_gabor_dictionary(length, freqs, sigmas, sr, phases=[0]):
    atoms = []
    for freq in freqs:
        for sigma in sigmas:
            for phase in phases:
                atom = generate_gabor_atom(length, freq, sigma, sr, phase)
                atoms.append(atom)
    return np.array(atoms).T  # Each column is an atom

def get_dictionary(chunk_size=2048, dictionary_size=10000, 
                   min_freq=30, max_freq=20000, sr=44100,
                   sigmas=[0.05, 0.1, 0.2, 0.5, 0.7, 1.0, 1.5]):
    freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), dictionary_size // len(sigmas))
    dictionary = create_gabor_dictionary(chunk_size, freqs, sigmas, sr)
    dictionary = dictionary.astype(np.float64)
    dictionary /= np.linalg.norm(dictionary, axis=0)
    return dictionary

# if __name__ == "__main__":
#     file_path = 'gospel.wav'
#     fs, audio_signal = wavfile.read(file_path)
#     audio_signal = audio_signal[:fs*5]
#     if audio_signal.ndim > 1:
#         audio_signal = audio_signal.mean(axis=1) 
#     audio_signal = audio_signal / np.max(np.abs(audio_signal))
#     chunk_size = int(44100//20)
#     dictionary_size = 10000
#     length = chunk_size  
#     sigmas = [0.05, 0.1, 0.2, 0.5, 0.7, 1.0, 1.5]  
#     freqs = np.linspace(30, 20000, dictionary_size//len(sigmas))
#     dictionary = create_gabor_dictionary(length, freqs, sigmas)
#     dictionary = dictionary.astype(np.float64)
    
#     audio_signal = audio_signal.astype(np.float64)
#     dictionary /= np.linalg.norm(dictionary, axis=0)

    
#     reconstructed_signal,chunks_info = process_in_chunks(audio_signal, 
#                                                          dictionary, 
#                                                          fs=fs, 
#                                                          chunk_size=chunk_size, 
#                                                          iterations=100)
#     reconstructed = reconstruct_from_chunks(chunks_info, dictionary, len(audio_signal))
    
#     timestampStr = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
#     sf.write(f"{file_path}_{timestampStr}.wav", reconstructed, fs)