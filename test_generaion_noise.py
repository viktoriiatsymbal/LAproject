import numpy as np
import scipy.io.wavfile as wav
import os
import random

def add_white_noise(signal, noise_level):
    noise = np.random.normal(0, noise_level * np.std(signal), size=signal.shape)
    return signal + noise

def add_gaussian_noise(signal, noise_level):
    noise = np.random.normal(0, noise_level, size=signal.shape)
    return signal + noise

def add_low_freq_noise(signal, noise_level, freq=50, sampling_rate=44100):
    t = np.arange(len(signal)) / sampling_rate
    noise = noise_level * np.sin(2 * np.pi * freq * t)
    return signal + noise

def add_impulse_noise(signal, noise_level, num_clicks=100):
    noisy_signal = signal.copy()
    indices = np.random.randint(0, len(signal), num_clicks)
    for idx in indices:
        noisy_signal[idx] += noise_level * np.max(signal) * np.random.choice([-1, 1])
    return noisy_signal

def process_audio_with_noises(input_file_path):
    sampling_rate, signal = wav.read(input_file_path)
    signal = signal.astype(np.float32)

    base_name = os.path.splitext(os.path.basename(input_file_path))[0]

    original_path = os.path.join(f'{base_name}_original.wav')
    wav.write(original_path, sampling_rate, signal.astype(np.int16))
    
    noises = {
        'white': add_white_noise,
        'gaussian': add_gaussian_noise,
        'lowfreq': add_low_freq_noise,
        'impulse': add_impulse_noise
    }
    
    output_paths = {'original': original_path}
    
    for noise_name, noise_func in noises.items():
        noise_level = random.uniform(0.01, 0.2)
        if noise_name == 'lowfreq':
            noisy_signal = noise_func(signal, noise_level, sampling_rate=sampling_rate)
        elif noise_name == 'impulse':
            num_clicks = int(len(signal) * random.uniform(0.001, 0.01))
            noisy_signal = noise_func(signal, noise_level, num_clicks=num_clicks)
        else:
            noisy_signal = noise_func(signal, noise_level)
        
        noisy_signal = np.clip(noisy_signal, -32768, 32767)
        path = os.path.join(f'{base_name}_{noise_name}.wav')
        wav.write(path, sampling_rate, noisy_signal.astype(np.int16))
        output_paths[noise_name] = path

    combined_signal = signal.copy()
    for noise_func in random.sample(list(noises.values()), k=3):
        noise_level = random.uniform(0.01, 0.2)
        if noise_func == add_low_freq_noise:
            combined_signal = noise_func(combined_signal, noise_level, sampling_rate=sampling_rate)
        elif noise_func == add_impulse_noise:
            num_clicks = int(len(signal) * random.uniform(0.001, 0.01))
            combined_signal = noise_func(combined_signal, noise_level, num_clicks=num_clicks)
        else:
            combined_signal = noise_func(combined_signal, noise_level)

    combined_signal = np.clip(combined_signal, -32768, 32767)
    combined_path = os.path.join(f'{base_name}_combined.wav')
    wav.write(combined_path, sampling_rate, combined_signal.astype(np.int16))
    output_paths['combined'] = combined_path

    return output_paths

output_files = process_audio_with_noises('hello.wav')
