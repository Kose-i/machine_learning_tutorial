import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.io.wavfile import write

duration = 4          # seconds
sampling_freq = 44100 # Hz
tone_freq = 784       # Hz

t = np.linspace(0, duration, duration*sampling_freq)
signal = np.sin(2 * np.pi * tone_freq * t)
noise = 0.5 * np.random.rand(duration * sampling_freq)
signal += noise

scaling_factor = 2**15 - 1
signal_normalized = signal / np.max(np.abs(signal))
signal_scaled = np.int16(signal_normalized* scaling_factor)
output_file = 'output/generated_audio.wav'
write(output_file, sampling_freq, signal_scaled)

size = 200
signal = signal[:size]
time_axis = np.linspace(0, 1000*size / sampling_freq, size)

plt.plot(time_axis, signal, color='black')
plt.xlabel('Time (milliseconds)')
plt.ylabel('Amplitude')
plt.title('Generated audio signal')
plt.show()

sampling_freq = 44100

def tone_synthesizer(freq, duration, amplitude=2**15-1):
    time_axis = np.linspace(0, duration, int(duration * sampling_freq))
    signal = amplitude * np.sin(2 * np.pi * freq * time_axis)
    return signal.astype(np.int16)

tone_map = {
  "A" :440,
  "A#":466,
  "B" :494,
  "C" :523,
  "C#":554,
  "D" :587,
  "D#":622,
  "E" :659,
  "F" :698,
  "F#":740,
  "G" :784,
  "G#":831
}

file_tone_signal = 'output/generated_tone_signal.wav'
synthesized_tone = tone_synthesizer(tone_map['F'], 3)
write(file_tone_signal, sampling_freq, synthesized_tone)
tone_sequence = [('G', 0.4), ('D', 0.5), ('F', 0.3), ('C', 0.6), ('A', 0.4)]

signal = np.array([], dtype=np.int16)
for tone_name, duration in tone_sequence:
    freq = tone_map[tone_name]
    synthesized_tone = tone_synthesizer(freq, duration)
    signal = np.append(signal, synthesized_tone, axis=0)
file_tone_sequence = 'generated_tone_sequence.wav'
write(file_tone_sequence, sampling_freq, signal)

from python_speech_features import mfcc, logfbank
sampling_freq, signal = wavfile.read('datasets/random_sound.wav')
signal = signal[:10000]

features_mfcc = mfcc(signal, sampling_freq)
print('MFCC:\nNumber of windows =', features_mfcc.shape[0])
print('Length of each feature =', features_mfcc.shape[1])

features_mfcc = features_mfcc.T
plt.matshow(features_mfcc)
plt.title('MFCC')
plt.show()

features_fb = logfbank(signal, sampling_freq)
print('Filter bank:\nNumber of windows =', features_fb.shape[0])
print('Length of each feature =', features_fb.shape[1])

features_fb = features_fb.T
plt.matshow(features_fb)
plt.title('Filter bank')
plt.show()

from hmmlearn import hmm

class ModelHMM(object):
    def __init__(self):
        self.models = []
        self.model = hmm.GaussianHMM(n_components=4, covariance_type='diag', n_iter=1000)
    def train(self, training_data):
        cur_model = self.model.fit(training_data)
        self.models.append(cur_model)
    def compute_score(self, input_data):
        return self.model.score(input_data)

def train_model(training_files):
    X = None
    for file in training_files:
        sampling_freq, signal = wavfile.read(file)
        features_mfcc = mfcc(signal, sampling_freq)
        if X is None:
            X = features_mfcc
        else:
            X = np.append(X, features_mfcc, axis=0)
    model = ModelHMM()
    model.train(X)
    return model
def build_models(wav_files):
    speech_models = []
    for label, files in wav_files.items():
        model = train_model(files[:-1])
        speech_models.append((model, label))
    return speech_models
def speech_recognition(speech_models, test_file):
    sampling_freq, signal = wavfile.read(test_file)
    features_mfcc = mfcc(signal, sampling_freq)
    scores = [model.compute_score(features_mfcc) for model, _ in speech_models]
    index = np.argmax(scores)
    return speech_models[index][1]
def run_tests(speech_models, wav_files):
    for original_label, files in wav_files.items():
        predicted_label = speech_recognition(speech_models, files[-1])
        print('\nOriginal:', original_label)
        print('Predicted:', predicted_label)

input_folder = 'datasets/data'

wav_files = {}
for root, dirs, files in os.walk(input_folder):
    files = [file for file in files if file.endswith('.wav')]
    if not files:
        continue
    label = files[0][:-6]
    wav_files[label] = [os.path.join(root, file) for file in files]
speech_models = build_models(wav_files)
run_tests(speech_models, wav_files)
