from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard

epo = 1
disp_num = 10	

encoding_dim = 32 

input_img = Input( shape = ( 28, 28, 1 ) )
x = Conv2D( 16,	( 3, 3 ), activation = 'relu', padding = 'same' )( input_img )
x = MaxPooling2D( ( 2, 2 ), padding = 'same' )( x )
x = Conv2D( 8, ( 3, 3 ), activation = 'relu', padding = 'same' )( x )
x = MaxPooling2D( ( 2, 2 ), padding = 'same' )( x )
x = Conv2D( 8, ( 3, 3 ), activation = 'relu', padding = 'same' )( x ) 					  
encoded = MaxPooling2D( ( 2, 2 ), padding = 'same' )( x )

x = Conv2D( 8, ( 3, 3 ), activation = 'relu', padding = 'same' )( encoded )
x = UpSampling2D( ( 2, 2 ) )( x )
x = Conv2D( 8, ( 3, 3 ), activation = 'relu', padding = 'same' )( x )
x = UpSampling2D( ( 2, 2 ) )( x )
x = Conv2D( 16, ( 3, 3 ), activation = 'relu', padding = 'same' )( x )
x = UpSampling2D( ( 2, 2 ) )( x )
decoded = Conv2D( 1, ( 3, 3 ), activation = 'sigmoid',
				  padding = 'same' )( x )


# 建立 autoencoder model
autoencoder = Model( input_img, decoded )

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
x_train = np.reshape( x_train, (len(x_train), 28, 28, 1 ))
x_test = np.reshape( x_test, ( len( x_test ),
						   28, 28, 1 ) ) 
print( x_train.shape[0], 'train samples' )						   
print( x_test.shape[0], 'test samples' )

# 開始 training
autoencoder.fit( x_train, x_train, 
				 epochs = epo,
				 batch_size = 128,
				 shuffle = True,
				 validation_data = ( x_test, x_test ),
				 callbacks = [TensorBoard
								(log_dir='/tmp/autoencoder')])

# 進行 testing 

decoded_imgs = autoencoder.predict( x_test )

#exit()

import matplotlib.pyplot as plt

n = disp_num # how many digits we will display
plt.figure( fig( 20, 4 ) )
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
					 