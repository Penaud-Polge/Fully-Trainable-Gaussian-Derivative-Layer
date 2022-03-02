from Layer import *

verb = 1

epochs = 2

num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

xIn = Input(shape = input_shape)
x = FTGDConvLayer(filters=16, kernel_size = (7,7), num_basis= 4, order=3, shared=False, cl_type='Full', trainability=[True, True, True], padding='same')(xIn)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = FTGDConvLayer(filters=32, kernel_size = (7,7), num_basis= 8, order=3, shared=False, cl_type='Full', trainability=[True, True, True], padding='same', strides=(2, 2))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = GlobalAveragePooling2D()(x)
x = Dense(num_classes, activation = 'softmax')(x)

model = Model(xIn,x)

model.compile(loss = tensorflow.keras.losses.categorical_crossentropy, optimizer  = tensorflow.keras.optimizers.Adam(learning_rate = 0.01), metrics=['accuracy'])

history = model.fit(x_train[:45000, :,:,:], y_train[:45000,:], batch_size = 45, epochs = epochs, validation_data=(x_train[45000:, :,:,:], y_train[45000:,:]), verbose = verb)

score = model.evaluate(x_test, y_test)

print('Test Accuracy : %.2f and Test Loss : %.3f' % (score[1]*100, score[0]))
