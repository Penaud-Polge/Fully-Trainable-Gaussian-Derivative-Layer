import tensorflow
import tensorflow.keras
import tensorflow.keras.backend as K
import numpy as np
import math
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *

from tensorflow.keras.models import *
from tensorflow.python.ops import nn


import glob
import skimage.io as io
import random





def hermitePolynomials(order, x, sigma):

    res = tensorflow.math.pow((np.sqrt(2)/sigma)* x,order)
    for i in range(1, (order//2)+1):
        term = (tensorflow.math.multiply(math.pow(-1, i)*math.factorial(order)/(math.factorial(i)*math.factorial(order - 2*i)), tensorflow.math.pow((tensorflow.math.divide(np.sqrt(2),sigma)),(order - 2*i)))) * tensorflow.math.pow(x, (order - 2*i))
        res = tensorflow.math.add(res, term)
    return res


def computeGaussianDerivative(order, x, sigma):
    
    hermitePart = tensorflow.math.multiply(tensorflow.math.pow(tensorflow.math.divide(-1,tensorflow.math.multiply(math.sqrt(2),sigma)),order), hermitePolynomials(order, x, sigma))
    gaussianPart = tensorflow.math.multiply(tensorflow.math.divide(1,tensorflow.math.multiply(sigma,np.sqrt(2*np.pi))), tensorflow.math.exp(- tensorflow.math.divide(tensorflow.math.pow(x, 2),(2*tensorflow.math.pow(sigma,2)))))
    
    gaussianDerivative = tensorflow.math.multiply(hermitePart, gaussianPart)
    return gaussianDerivative


def computeGaussianBasis(size, order, sigmas, centroids, thetas):

    kernels = []
    [x,y] = tensorflow.meshgrid(range(-int(size[0]/2), int(size[0]/2) + 1), range(-int(size[1]/2), int(size[1]/2) + 1))
    x = tensorflow.cast(x, tensorflow.float32)
    y = tensorflow.cast(y, tensorflow.float32)
    counter = 0
    for i in range(order+1):
        for j in range(i+1):

            u = tensorflow.math.add(tensorflow.multiply(tensorflow.math.cos(thetas[counter]), x), tensorflow.math.multiply(tensorflow.math.sin(thetas[counter]), y))
            v = tensorflow.math.add(tensorflow.multiply(-tensorflow.math.sin(thetas[counter]), x), tensorflow.math.multiply(tensorflow.math.cos(thetas[counter]), y))
                
            dGaussx = computeGaussianDerivative(j, tensorflow.math.add(u, - centroids[counter, 0]), sigmas[counter, 0])
            dGaussy = computeGaussianDerivative(i-j, tensorflow.math.add(v, - centroids[counter, 1]), sigmas[counter, 1])
            
            dGauss = tensorflow.math.multiply(dGaussx, dGaussy)
            kernels.append(tensorflow.expand_dims(dGauss, -1))
            counter += 1
    return tensorflow.stack(kernels, axis = -1)


def getBasis(size, numBasis, order, sigmas, centroids, thetas):
    
    basis = []
    for i in range(numBasis):
        basis.append(computeGaussianBasis(size, order, sigmas[i,:,:], centroids[i,:,:], thetas[i,:]))
        
    return tensorflow.stack(basis, axis = 0)

def initWeights(inputChannels, outputChannels, numBasis, order, shared = True, clType = 'Full'):
    
    numFiltersPerBasis = (order + 1)*(order + 2)/2
    if clType == 'Full':
        if shared:
            weights = tensorflow.Variable(initial_value = tensorflow.random.uniform(shape = (int(inputChannels),  int(outputChannels/numBasis), int(numFiltersPerBasis)), minval = -1, maxval = 1, dtype = 'float'), name = 'clWeights', trainable = True)
        else:
            weights = tensorflow.Variable(initial_value = tensorflow.random.uniform(shape = (int(numBasis), int(inputChannels),   int(outputChannels/numBasis), int(numFiltersPerBasis)), minval = -1, maxval = 1, dtype = 'float'), name = 'clWeights', trainable = True)
    elif clType == 'Separated':
        if shared:
            weights1 = tensorflow.Variable(initial_value = tensorflow.random.uniform(shape = (int(inputChannels), int(numFiltersPerBasis)), minval = -1, maxval = 1, dtype = 'float'), name = 'clWeights1', trainable = True)
            weights2 = tensorflow.Variable(initial_value = tensorflow.random.uniform(shape = (1,1, int(numFiltersPerBasis), int(outputChannels/numBasis)), minval = -1, maxval = 1, dtype = 'float'), name = 'clWeights2', trainable = True)
            weights = [weights1, weights2]
        else:
            weights1 = tensorflow.Variable(initial_value = tensorflow.random.uniform(shape = (int(numBasis),  int(inputChannels), int(numFiltersPerBasis)), minval = -1, maxval = 1, dtype = 'float'), name = 'clWeights1', trainable = True)
            weights2 = tensorflow.Variable(initial_value = tensorflow.random.uniform(shape = (int(numBasis),  1,1, int(numFiltersPerBasis), int(outputChannels/numBasis)), minval = -1, maxval = 1, dtype = 'float'), name = 'clWeights2', trainable = True)
            weights = [weights1, weights2]
    else:
        weights = None
        print("Please use Full or Separated as clType")
        
    return weights


def initGaussianParameters(numBasis, order, random, trainability, sigma_init, mu_init, theta_init):

    numFiltersPerBasis = (order + 1)*(order + 2)/2
    if random:
        sigmas = tensorflow.Variable(initial_value = tensorflow.random.uniform(shape = (int(numBasis), int(numFiltersPerBasis), 2), minval = 0.5, maxval = 2, dtype = 'float'),  name = 'sigmas', trainable = trainability[0], constraint=tensorflow.keras.constraints.MinMaxNorm(min_value=0.2, max_value=100, rate=1.0, axis=[0, 1, 2]))
        centroids = tensorflow.Variable(initial_value = tensorflow.random.uniform(shape = (int(numBasis), int(numFiltersPerBasis), 2), minval = -1, maxval = 1, dtype = 'float'),  name = 'centroids', trainable = trainability[1])
        thetas = tensorflow.Variable(initial_value = tensorflow.random.uniform(shape = (int(numBasis), int(numFiltersPerBasis)), minval = -math.pi, maxval = math.pi, dtype = 'float'),  name = 'thetas', trainable = trainability[2])
        
    else:
        sigmas = tensorflow.Variable(initial_value = tensorflow.random.uniform(shape = (int(numBasis), int(numFiltersPerBasis), 2), minval = sigma_init, maxval = sigma_init, dtype = 'float'),  name = 'sigmas', trainable = trainability[0], constraint=tensorflow.keras.constraints.MinMaxNorm(min_value=0.2, max_value=100, rate=1.0, axis=[0, 1, 2]))
        centroids = tensorflow.Variable(initial_value = tensorflow.random.uniform(shape = (int(numBasis), int(numFiltersPerBasis), 2), minval = mu_init, maxval = mu_init, dtype = 'float'),  name = 'centroids', trainable = trainability[1])
        thetas = tensorflow.Variable(initial_value = tensorflow.random.uniform(shape = (int(numBasis), int(numFiltersPerBasis)), minval = theta_init, maxval = theta_init, dtype = 'float'),  name = 'thetas', trainable = trainability[2])

    return sigmas, centroids, thetas

def computeOutput(basis, inputs, weights, numBasis, inputChannels, outputChannels, clType, shared, paddingMode, stride = (1,1)):


    if clType == 'Full':

        if shared:

            consideredBasis = basis[0,:,:,:,:]
            consideredBasis = tensorflow.expand_dims(consideredBasis, axis = -2)
            consideredBasis = tensorflow.tile(consideredBasis, [1,1, inputChannels, int(outputChannels/numBasis), 1])
            filters = tensorflow.multiply(consideredBasis, weights)
            filters = tensorflow.reduce_sum(filters, axis = -1)
            outputs = K.conv2d(inputs, filters, strides = stride, padding = paddingMode)

            for i in range(1, numBasis):

                consideredBasis = basis[i,:,:,:,:]
                consideredBasis = tensorflow.expand_dims(consideredBasis, axis = -2)
                consideredBasis = tensorflow.tile(consideredBasis, [1,1, inputChannels, int(outputChannels/numBasis), 1])
                filters = tensorflow.multiply(consideredBasis, weights)
                filters = tensorflow.reduce_sum(filters, axis = -1)
                res = K.conv2d(inputs, filters, strides = stride, padding = paddingMode)
                
                outputs = tensorflow.concat([outputs, res], axis = -1)
        
        else:

            consideredBasis = basis[0,:,:,:,:]
            consideredBasis = tensorflow.expand_dims(consideredBasis, axis = -2)
            consideredBasis = tensorflow.tile(consideredBasis, [1,1, inputChannels, int(outputChannels/numBasis), 1])
            filters = tensorflow.multiply(consideredBasis, weights[0,:,:,:])
            filters = tensorflow.reduce_sum(filters, axis = -1)
            outputs = K.conv2d(inputs, filters, strides = stride, padding = paddingMode)

            for i in range(1, numBasis):

                consideredBasis = basis[i,:,:,:,:]
                consideredBasis = tensorflow.expand_dims(consideredBasis, axis = -2)
                consideredBasis = tensorflow.tile(consideredBasis, [1,1, inputChannels, int(outputChannels/numBasis), 1])
                filters = tensorflow.multiply(consideredBasis, weights[i,:,:,:])
                filters = tensorflow.reduce_sum(filters, axis = -1)
                res = K.conv2d(inputs, filters, strides = stride, padding = paddingMode)

                outputs = tensorflow.concat([outputs, res], axis = -1)

    elif clType == 'Separated':

        if shared:

            consideredBasis = basis[0,:,:,:,:]
            consideredBasis = tensorflow.tile(consideredBasis, [1,1,inputChannels,1])
            filters1 = tensorflow.multiply(consideredBasis, weights[0])
            res1 = K.conv2d(inputs, filters1, strides = stride, padding = paddingMode)
            outputs = K.conv2d(res1, weights[1], padding = paddingMode)

            for i in range(1, numBasis):

                consideredBasis = basis[i,:,:,:,:]
                consideredBasis = tensorflow.tile(consideredBasis, [1,1,inputChannels,1])
                filters1 = tensorflow.multiply(consideredBasis, weights[0])
                res1 = K.conv2d(inputs, filters1, strides = stride, padding = paddingMode)
                res2 = K.conv2d(res1, weights[1], padding = paddingMode)

                outputs = tensorflow.concat([outputs, res2], axis = -1)
                
        else:

            consideredBasis = basis[0,:,:,:,:]
            consideredBasis = tensorflow.tile(consideredBasis, [1,1,inputChannels,1])
            filters1 = tensorflow.multiply(consideredBasis, weights[0][0,:,:])
            res1 = K.conv2d(inputs, filters1, strides = stride, padding = paddingMode)
            outputs = K.conv2d(res1, weights[1][0,:,:,:,:], padding = paddingMode)

            for i in range(1, numBasis):

                consideredBasis = basis[i,:,:,:,:]
                consideredBasis = tensorflow.tile(consideredBasis, [1,1,inputChannels,1])
                filters1 = tensorflow.multiply(consideredBasis, weights[0][i,:,:])
                res1 = K.conv2d(inputs, filters1, strides = stride, padding = paddingMode)
                res2 = K.conv2d(res1, weights[1][i,:,:,:,:], padding = paddingMode)

                outputs = tensorflow.concat([outputs, res2], axis = -1)
    else:
        outputs = None
        print("Please use Full or Separated for clType")
    
    return outputs