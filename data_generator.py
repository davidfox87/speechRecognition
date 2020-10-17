import numpy as np
from util import log_specgram, load_audio, calc_num_freqs, text_to_int_sequence
from sklearn.preprocessing import StandardScaler
import os


class AudioGenerator:

	def __init__(self, window=20, max_freq=8000, batch_size=20,
				 descr_file='', sort_by_duration=True):
		self.feat_dim = calc_num_freqs(window, max_freq)
		self.audio_paths = None
		self.texts = None
		self.durations = None
		self.batch_size = batch_size
		self.max_freq = max_freq
		self.index = 0
		self.sort_by_duration = sort_by_duration
		s = descr_file.split('/')[:4]
		self.prefix = os.path.join(*s)
		if descr_file is not None:
			self.load_data(descr_file)

	def load_data(self, descr_file='LibriSpeech/dev-clean/84/121123/84-121123.trans.txt'):
		'''
		get the audio file names and the utterances from the metadata file
		:param descr_file: meta data file that has the information about audio file names and spoken words
		:return:
		'''

		audio_paths = []
		texts = []
		with open(descr_file) as ft:
			for line in ft:
				fileid_text, utterance = line.strip().split(" ", 1)

				audio_paths.append(fileid_text)
				texts.append(utterance)

		self.audio_paths = audio_paths
		self.texts = texts
		self.durations = [len(s) for s in self.texts]

		# we sort so that we start learning the easiest words first
		if self.sort_by_duration:
			self.sort_data()

	def sort_data(self):
		'''
		sort files based on length of utterances
		This way the neural network gets a chance to learn the easiest
		words first
		:return:
		'''
		p = np.argsort(self.durations).tolist()
		self.audio_paths = [self.audio_paths[i] for i in p]
		self.durations = [self.durations[i] for i in p]
		self.texts = [self.texts[i] for i in p]

	def get_spectrogram(self, file):
		'''
		gets the spectrogram given an audio filename
		:param file:
		:return: the spectrogram
		'''
		# load audio from file
		audio, sf = load_audio(file, prefix=self.prefix)

		# get spectrogram
		spectrogram = log_specgram(audio, sf)

		return spectrogram

	def standardize(self, x):
		'''
		standardizes the spectrogram features
		:param x:
		:return:
		'''
		scaler = StandardScaler()
		return scaler.fit_transform(x)

	def get_batch(self):
		'''
		  'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		:return:
		'''

		index = self.index

		# get file names of the batch
		audio_samples = self.audio_paths[index * self.batch_size: (index + 1) * self.batch_size]

		# generate data	(standardize each spectrogram)
		features = [self.standardize(self.get_spectrogram(a)) for a in audio_samples]

		# calculate necessary sizes
		max_length = max([features[i].shape[0]
						  for i in range(self.batch_size)])

		max_string_length = max([len(self.texts[index + i])
								 for i in range(self.batch_size)])

		# initialize the arrays
		X_data = np.zeros([self.batch_size, max_length,
						   self.feat_dim])

		labels = np.ones([self.batch_size, max_string_length]) * 28
		input_length = np.zeros([self.batch_size, 1])
		label_length = np.zeros([self.batch_size, 1])

		for i in range(0, self.batch_size):
			# calculate X_data & input_length
			feat = features[i]
			input_length[i] = feat.shape[0]
			X_data[i, :feat.shape[0], :] = feat

			# calculate labels & label_length
			label = np.array(text_to_int_sequence(self.texts[index + i]))
			labels[i, :len(label)] = label
			label_length[i] = len(label)

		# return the arrays
		outputs = {'ctc': np.zeros([self.batch_size])}
		inputs = {'the_input': X_data,
				  'the_labels': labels,
				  'input_length': input_length,
				  'label_length': label_length
				  }

		return inputs, outputs

	def next(self):
		""" Obtain a batch of training data
		"""
		# keep looping until we reach our batch size
		while True:
			ret = self.get_batch()
			self.index += self.batch_size
			if self.index >= len(self.texts) - self.batch_size:
				self.index = 0
			yield ret
