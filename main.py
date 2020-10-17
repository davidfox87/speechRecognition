
from model import cnn_lstm_model, cnn_rnn_model, train_model
from data_generator import AudioGenerator
from glob import glob
import numpy as np
import _pickle as pickle
import seaborn as sns
import matplotlib.pyplot as plt
from keras import backend as K
from util import int_sequence_to_text, text_to_int_sequence
import re

def get_predictions(datagen, index, input_to_softmax, model_path):
	'''

	:param datagen: an audiogenerator corresponding to train or validation
	:param index:
	:param input_to_softmax:
	:param model_path:
	:return:
	'''
	transcr = datagen.texts[index]
	audio_path = datagen.audio_paths[index]
	data_point = datagen.standardize(datagen.get_spectrogram(audio_path))

	# Obtain and decode the acoustic model's predictions
	input_to_softmax.load_weights(model_path)
	prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
	output_length = [input_to_softmax.output_length(data_point.shape[0])]
	pred_ints = (K.eval(K.ctc_decode(
		prediction, output_length)[0][0])).flatten().tolist()

	pred_ints = np.array(pred_ints)
	pred_ints = pred_ints[pred_ints > 0]

	print('-' * 80)
	print('True transcription:\n' + '\n' + transcr)
	print('-' * 80)
	print('Predicted transcription:\n' + '\n' + ''.join(int_sequence_to_text(pred_ints)))
	print('-' * 80)


'''
build a neural net that takes frequency components for each time window in a spectrogram
and predicts character sequence

neural network will process each frame of the spectrogram

length of x != length of y
use CTC = connectionist temporal classification
'''
if __name__ == "__main__":
	# datasets
	partition = {'train': 'LibriSpeech/dev-clean/84/121123/84-121123.trans.txt',
				 'validation': 'LibriSpeech/dev-clean/84/121550/84-121550.trans.txt'}

	# Generators
	training_generator = AudioGenerator(descr_file=partition['train'], batch_size=20)
	validation_generator = AudioGenerator(descr_file=partition['validation'], batch_size=20)

	# get this model working first and then use lstm
	model = cnn_rnn_model(input_dim=161,
				   filters=200,
				   kernel_size=11,
				   conv_stride=2,
				   conv_border_mode='valid',
				   units=200)

	train_model(input_to_softmax=model,
				pickle_path='model_0.pickle',
				train_generator=training_generator,
				validation_generator=validation_generator,
				save_model_path='model_0.h5')



	# look at training and validation curves

	# all_pickles = sorted(glob("results/*.pickle"))
	# # Extract the name of each model
	# model_names = [item[8:-7] for item in all_pickles]
	#
	# valid_loss = [pickle.load(open(i, "rb"))['val_loss'] for i in all_pickles]
	# train_loss = [pickle.load(open(i, "rb"))['loss'] for i in all_pickles]
	#
	# num_epochs = [len(valid_loss[i]) for i in range(len(valid_loss))]
	#
	# fig = plt.figure(figsize=(16, 5))
	#
	# # Plot the training loss vs. epoch for each model
	# ax1 = fig.add_subplot(121)
	# for i in range(len(all_pickles)):
	# 	ax1.plot(np.linspace(1, num_epochs[i], num_epochs[i]),
	# 			 train_loss[i], label=model_names[i])
	# # Clean up the plot
	# ax1.legend()
	# ax1.set_xlim([1, max(num_epochs)])
	# plt.xlabel('Epoch')
	# plt.ylabel('Training Loss')
	#
	# ax2 = fig.add_subplot(122)
	# for i in range(len(all_pickles)):
	# 	ax2.plot(np.linspace(1, num_epochs[i], num_epochs[i]),
	# 			 valid_loss[i], label=model_names[i])
	# # Clean up the plot
	# ax2.legend()
	# ax2.set_xlim([1, max(num_epochs)])
	# plt.xlabel('Epoch')
	# plt.ylabel('Validation Loss')
	# plt.show()


	get_predictions(training_generator, 0,
					input_to_softmax=model,
                	model_path='results/model_0.h5')