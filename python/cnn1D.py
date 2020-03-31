import keras
from keras import layers
from keras import optimizers

# define convolutional layer

#kernel size
filters =4
kernel_size = 4

convolution_1d_layer = keras.layers.convolutional.Conv1D(filters, kernel_size, strides=1, padding='same', input_shape=(500, 1), activation="relu", name="convolution_1d_layer")

# 定义最大化池化层
max_pooling_layer = keras.layers.MaxPooling1D(pool_size=2, strides=2,padding="same", name="max_pooling_layer")

# reshape layer
reshape_layer = keras.layers.core.Flatten(name="reshape_layer")

# dropout layer
dropout_layer=keras.layers.Dropout(0.5,name="dropout_layer")

# full connect layer
full_connect_layer = keras.layers.Dense(128, activation="relu", name="full_connect_layer")

model = keras.Sequential()
model.add(convolution_1d_layer)
model.add(max_pooling_layer)
model.add(reshape_layer)
model.add(dropout_layer)
model.add(full_connect_layer)
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(
              optimizer= optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
              loss='binary_crossentropy',
              metrics=['accuracy']
              )
model.save('RunCnn.model')
print(model.summary())
