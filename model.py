import keras
from keras.models import Model
from keras.layers import Input, Activation, Conv1D
from keras.layers import Dense, Flatten, BatchNormalization, LSTM, TimeDistributed, Dropout, MaxPooling1D, SimpleRNN
from keras import backend as K
from keras.layers import (Input, Lambda)
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import os
import pickle


def cnn_lstm_model(input_dim, filters, kernel_size, conv_stride,
	conv_border_mode, units, n_outputs=29):
	""" Build a recurrent + convolutional network for speech
	"""
	# Main acoustic input
	input_data = Input(name='the_input', shape=(None, input_dim))

	# Add convolutional layers
	# TimeDistributed applies a layer to every temporal slice of an input
	# i.e. each time slice of the input spectrogram
	x = Conv1D(filters, kernel_size,
					 strides=conv_stride,
					 padding=conv_border_mode,
					 activation='relu',
					 name='conv1d')(input_data)

	# Add batch normalization...not clear what this does yet
	x = BatchNormalization(name='bn_conv_1d')(x)

	x = LSTM(units)(x)

	x = TimeDistributed(Dense(n_outputs))(x)

	# Add softmax activation layer
	predictions = Activation('softmax', name='softmax')(x)
	# Specify the model
	model = Model(inputs=input_data, outputs=predictions)
	model.output_length = lambda x: cnn_output_length(
		x, kernel_size, conv_border_mode, conv_stride)

	print(model.summary())
	return x





def cnn_output_length(input_length, filter_size, border_mode, stride,
					   dilation=1):
	""" Compute the length of the output sequence after 1D convolution along
		time. Note that this function is in line with the function used in
		Convolution1D class from Keras.
	Params:
		input_length (int): Length of the input sequence.
		filter_size (int): Width of the convolution kernel.
		border_mode (str): Only support `same` or `valid`.
		stride (int): Stride size used in 1D convolution.
		dilation (int)
	"""
	if input_length is None:
		return None
	assert border_mode in {'same', 'valid'}
	dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
	if border_mode == 'same':
		output_length = input_length
	elif border_mode == 'valid':
		output_length = input_length - dilated_filter_size + 1
	return (output_length + stride - 1) // stride

def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
	conv_border_mode, units, output_dim=29):
	""" Build a recurrent + convolutional network for speech
	"""
	# Main acoustic input
	input_data = Input(name='the_input', shape=(None, input_dim))
	# Add convolutional layer
	conv_1d = Conv1D(filters, kernel_size,
					 strides=conv_stride,
					 padding=conv_border_mode,
					 activation='relu',
					 name='conv1d')(input_data)
	# Add batch normalization
	bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
	# Add a recurrent layer
	simp_rnn = SimpleRNN(units, activation='relu',
		return_sequences=True, name='rnn')(bn_cnn)
	# Add batch normalization
	bn_rnn = BatchNormalization()(simp_rnn)
	# Add a TimeDistributed(Dense(output_dim)) layer
	time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
	# Add softmax activation layer
	y_pred = Activation('softmax', name='softmax')(time_dense)
	# Specify the model
	model = Model(inputs=input_data, outputs=y_pred)
	model.output_length = lambda x: cnn_output_length(
		x, kernel_size, conv_border_mode, conv_stride)
	print(model.summary())
	return model


def ctc_lambda_func(args):
	y_pred, labels, input_length, label_length = args
	return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def add_ctc_loss(input_to_softmax):
	the_labels = Input(name='the_labels', shape=(None,), dtype='float32')
	input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
	label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
	output_lengths = Lambda(input_to_softmax.output_length)(input_lengths)
	# CTC loss is implemented in a lambda layer
	loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
		[input_to_softmax.output, the_labels, output_lengths, label_lengths])
	model = Model(
		inputs=[input_to_softmax.input, the_labels, input_lengths, label_lengths],
		outputs=loss_out)
	return model



def train_model(input_to_softmax,
				pickle_path,
				save_model_path,
				train_generator,
				validation_generator,
				batch_size=1,
				epochs=5,
				verbose=1):

	# calculate steps_per_epoch
	num_train_examples = len(train_generator.audio_paths)
	steps_per_epoch = num_train_examples // batch_size

	# calculate validation_steps
	num_valid_samples = len(validation_generator.audio_paths)
	validation_steps = num_valid_samples // batch_size

	# add CTC loss to the NN specified in input_to_softmax
	model = add_ctc_loss(input_to_softmax)

	optimizer = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

	# CTC loss is implemented elsewhere, so use a dummy lambda function for the loss
	model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

	# make results/ directory, if necessary
	if not os.path.exists('results'):
		os.makedirs('results')

	# add checkpointer
	checkpointer = ModelCheckpoint(filepath='results/' + save_model_path, verbose=0)

	# train the model
	hist = model.fit(train_generator.next(),
					 epochs=epochs,
					 callbacks=[checkpointer],
					 steps_per_epoch=steps_per_epoch,
					 verbose=verbose)

	# save model loss
	with open('results/' + pickle_path, 'wb') as f:
		pickle.dump(hist.history, f)





