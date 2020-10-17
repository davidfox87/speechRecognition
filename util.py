from scipy.signal import spectrogram
import numpy as np
import soundfile as sf
import os


def calc_num_freqs(window_size=20, max_freq=8000):
	'''
	calculates number of features (frequencies) in spectrogram
	:param window_size:
	:param max_freq:
	:return:
	'''
	return int(0.001 * window_size * max_freq) + 1


# typically 20 ms window
# convert signal in window to frequency domain using fft and then take log
# what's the amount of power in that sine wave that makes up that original part of
# the signal
# concatenate frames from adjacent windows
def log_specgram(audio, sample_rate, window_size=20,
				 step_size=10, eps=1e-10):
	nperseg = int(round(window_size * sample_rate / 1e3))
	noverlap = int(round(step_size * sample_rate / 1e3))
	freqs, times, spec = spectrogram(audio,
									 fs=sample_rate,
									 window='hann',
									 nperseg=nperseg,
									 noverlap=noverlap,
									 detrend=False)

	return np.log(spec.T.astype(np.float32) + eps)


def load_audio(name, prefix='', ext_audio='.flac'):
	file_audio = os.path.join(prefix + '/', name + '.flac')

	with sf.SoundFile(file_audio) as sound_file:
		audio = sound_file.read(dtype='float32')
		sample_rate = sound_file.samplerate
		if audio.ndim >= 2:
			audio = np.mean(audio, 1)

	return audio, sample_rate


"""
Defines two dictionaries for converting 
between text and integer sequences.
"""

char_map_str = """
' 0
<SPACE> 1
a 2
b 3
c 4
d 5
e 6
f 7
g 8
h 9
i 10
j 11
k 12
l 13
m 14
n 15
o 16
p 17
q 18
r 19
s 20
t 21
u 22
v 23
w 24
x 25
y 26
z 27
"""
# the "blank" character is mapped to 28

char_map = {}
index_map = {}
for line in char_map_str.strip().split('\n'):
	ch, index = line.split()
	char_map[ch] = int(index)
	index_map[int(index)] = ch


def text_to_int_sequence(text):
	""" Convert text to an integer sequence """
	int_sequence = []
	for c in text:
		if c == ' ':
			ch = char_map['<SPACE>']
		else:
			ch = char_map[c.lower()]
		int_sequence.append(ch)
	return int_sequence


def int_sequence_to_text(int_sequence):
	""" Convert an integer sequence to text """
	text = []
	for c in int_sequence:
		if c == 1:
			ch = ' '
		else:
			ch = index_map[c]
		text.append(ch)
	return text
