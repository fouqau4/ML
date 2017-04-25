from keras.layers import Input, Dense
from keras.models import Model

epo = 50
disp_num = 10	

encoding_dim = 32 

input_img = Input( shape = ( 784, ) )
encoded = Dense( encoding_dim, activation = 'relu' )( input_img )
decoded = Dense( 784, activation = 'sigmoid' )( encoded )

# 建立 autoencoder model
autoencoder = Model( input_img, decoded )
# 建立 encoder model
encoder = Model( input_img, encoded )

encoded_input = Input( shape = ( encoding_dim, ) )

# autoencoder 有三個 layer
# autoencoder.layers[0] = input layer
# autoencoder.layers[1] = hidden layer
# autoencoder.layers[2] = output layer
# autoencoder.layers[-1] = autoencoder.layers[2]
decoder_layer = autoencoder.layers[-1]

# 建立 decoder model
# decoder_layer( encoded_input ) : 設定 input neuron 數量
decoder = Model( encoded_input, decoder_layer( encoded_input ) )

# 設定 training model
autoencoder.compile( optimizer = 'adadelta',
					 loss = 'binary_crossentropy' )

from keras.datasets import mnist

import numpy as np

# type( mnist.load_data() ) = <class 'tuple'>
# mnist.load_data = ( x_train, y_train ), ( x_test, y_test )
# x_train = 60000x28x28
# x_test = 10000x28x28
( x_train, _ ), ( x_test, _ ) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# x_train.shape = ( 60000, 28, 28 )
# x_train.shape[1:] = ( 28, 28 )
x_train = x_train.reshape( ( len( x_train ),
							 np.prod( x_train.shape[1:] ) ) )
x_test = x_test.reshape( ( len( x_test ),
						   np.prod( x_test.shape[1:] ) ) )
print( x_train.shape[0], 'train samples' )						   
print( x_test.shape[0], 'test samples' )

# 開始 training
autoencoder.fit( x_train, x_train, 
				 epochs = epo,
				 batch_size = 256,
				 shuffle = True,
				 validation_data = ( x_test, x_test ) )

# 進行 testing 
encoded_imgs = encoder.predict( x_test )
decoded_imgs = decoder.predict( encoded_imgs )

#exit()

import matplotlib.pyplot as plt

n = disp_num # how many digits we will display
plt.figure( figsize = ( 20, 4 ) )
for i in range( n ):
	# display original
	ax = plt.subplot( 2, n, i + 1 )
	plt.imshow( x_test[i].reshape( 28, 28 ) )
	plt.gray()
	ax.get_xaxis().set_visible( False )
	ax.get_yaxis().set_visible( False )
	
	# display reconstruction
	ax = plt.subplot( 2, n, i + 1 + n )
	plt.imshow( decoded_imgs[i].reshape( 28, 28 ) )
	plt.gray()
	ax.get_xaxis().set_visible( False )
	ax.get_yaxis().set_visible( False )
plt.show()
					 