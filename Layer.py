from Utils import *


class FTGDConvLayer(tensorflow.keras.layers.Layer):

    
    def __init__(self, filters, kernel_size,  num_basis, order, shared, cl_type, trainability, padding, sigma_init = 1, mu_init = 0, theta_init = 0, strides = (1,1), random_init = True, use_bias = False, **kwargs):

        super(FTGDConvLayer, self).__init__()
        self.num_filters = filters
        self.filter_size = kernel_size
        self.numBasis = num_basis
        self.order = order
        self.shared = shared
        self.clType = cl_type
        self.trainability = trainability
        self.paddingMode = padding
        self.stride = strides
        self.randomInit = random_init
        self.sigma_init = sigma_init
        self.mu_init =mu_init
        self.theta_init = theta_init
        self.use_bias = use_bias

    def build(self, input_shape):
        
        self.sigmas, self.centroids, self.thetas = initGaussianParameters(self.numBasis, self.order, self.randomInit, self.trainability, self.sigma_init, self.mu_init, self.theta_init)

        if self.use_bias:
            self.bias = tensorflow.Variable(initial_value = tensorflow.zeros(shape = (self.num_filters,), dtype = 'float'),  name = 'bias', trainable = True)
        else:
            self.bias = None
            
        self.clWeights = initWeights(input_shape[-1], self.num_filters, self.numBasis, self.order, self.shared, self.clType)
        self.inputChannels = input_shape[-1]

    def call(self, inputs):

        #outputs = computeOutput(self.gaussianBasis, inputs, self.clWeights, self.numBasis, self.inputChannels, self.outputChannels, self.clType, self.shared, self.paddingMode)
        outputs = computeOutput(getBasis(self.filter_size, self.numBasis, self.order, self.sigmas, self.centroids, self.thetas), inputs, self.clWeights, self.numBasis, self.inputChannels, self.num_filters, self.clType, self.shared, self.paddingMode, self.stride)

        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')
            
        return outputs


    def get_config(self):
        config = super(FTGDConvLayer, self).get_config()
        config.update({
            "num_filters":self.num_filters,
            "kernel_size":self.filter_size,
            'num_basis':self.numBasis,
            'order':self.order,
            'shared':self.shared,
            'cl_type':self.clType,
            'trainability':self.trainability,
            'stride':self.stride,
            'random_init':self.randomInit,
            'padding':self.paddingMode,
            'sigma_init':self.sigma_init,
            'mu_init':self.mu_init,
            'theta_init':self.theta_init,
            'use_bias':self.use_bias
        })
        return config