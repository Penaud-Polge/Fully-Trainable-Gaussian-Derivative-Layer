# FULLY TRAINABLE GAUSSIAN DERIVATIVE CONVOLUTIONAL LAYER

## What is it ?

This github corresponds to the Tensorflow implementation of the article

Valentin Penaud--Polge, Santiago Velasco-Forero, Jesus Angulo,
Fully Trainable Gaussian Derivative Convolutional Layer,
29th IEEE International Conference on Image Processing, 2022.

Please cite this reference if you use this code for your paper !

Roughly speaking, the particularity of this layer comes from its filters. 
Each filter is a linear combination of several anisotropic, shifted and rotated
Gaussian Derivative kernels.

## Recquirement

Tensorflow 2 is recquired to use this code.

## Some explanations

You will find two python files in this repository: The source code and a notebook showing a simple example showing how to use the proposed layer.

Here are the parameters of the layer:

 * filters   - int             - the number of filters.

 * kernel_size  - tuple of int    - kernel size used.

 * num_basis    - int             - number of bases used in the layer.

 * order        - int             - maximal order of derivation of the Gaussian Derivative kernels.
                                 
 * separated    - boolean         - indicates if the linear combination should be separated or not.
                                 
 * trainability - list of boolean - indicates if the Gaussian parameters should be trainable or not.
 
                           example : trainability = [True, False, True]
                           -> scales will be trainable
                           -> shifts won't be trainable
                           -> orientations will be trainable
                                        
 * padding      - string          - type of padding

 * sigma_init   - float           - initialization value of the scales (if random_init = False)
                                 
 * mu_init      - float           - initialization value of the shifts (if random_init = False)
                                 
 * theta_init   - float           - initialization value of the orientation (if random_init = False)
 
                            example : if sigma_init = 1.5, 
                            trainability[0] = False and 
                            random_init = False then
                            the Gaussian Derivative kernels 
                            will all have constant scales 
                            of value 1.5.
                                           
 * strides      - tuple of int    - value of the stride

 * random_init  - boolean         - whether or not the initialization should be random. If false, sigma_init, mu_init and theta_init are used.
                                 
 * use_bias     - boolean         - whether a bias should be used or not.


A little precision about the attribute "deployed" and the functions "deploy" and "to_train":

Using "deploy" once your layer is trained allows to freeze the Gaussian kernels. Otherwise the
layer will compute the Gaussian kernels every time you will use it, which is useless if the layer has already been trained.

Nevertheless, if you deployed your layers and afterwards you want to train them a little bit more, you can use the function to_train which will set back the attribute "deployed" to False. 














