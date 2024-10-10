import numpy as np
from tabulate import tabulate # For printing model summaries. This is useful for debugging but not strictly necessary.
from typing import Dict, Any, List, Tuple

activation_functions = {
    'none': lambda x: x,
    'relu': lambda x: x * (x > 0),
    'softmax': lambda x: softmax(x),
    'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-x)),
    'tanh': lambda x: (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
}

def softmax(x):
    """Compute softmax along axis 1."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)[:,np.newaxis]

class Layer:
    """
    This class is the base class for all layers of the `layers` module. 

    :param name: The name of the layer.
    :type name: str
        
    """

    def __init__(self, name: str):
        # Attributes        
        self.name = name
        self.weights = {}
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        This method is the base method for the transformation executed by a layer. It provides the mapping from x to y that makes each layer unique.

        :param x: The data that shall be transformed. It is assumed to be a tensor whose first axis enumerates instances. The other axes depend on the specific requirements of the current layer.
        :type x: numpy.ndarray

        :returns y (numpy.ndarray): The output of the transformation of this layers.
        """

        pass

    def load_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """This method loads the weights for the current layer.
        
        :param weights: A dictionary whose keys are name for weights and whose values are the corresponding weights. The naming pattern used here is 'kernel:kernel_index' and 'bias:bias_index'. So, for example, 'kernel:0' or 'bias:0' for the layer's first kernel matrix and bias vector.
        :type weights: Dict[str, numpy.ndarray]. """
        self.weights = weights

    # This function needs to be overridden by all layers that use weights.
    def weight_count(self) -> int:
        """This method counts the number of weights of the current layer.
        
        :returns count (int): The total count of weights."""

        # Compute weight count
        weight_count = 0
        for name, weight in self.weights.items():
            weight_count += np.prod(weight.shape)

        # Output
        return weight_count

class Flatten(Layer):
    """
    This layer flattens any input from it original shape (instance count, ...) to (instance_count, total_dimension_count). For example, if an input has shape (instance_count, height, width, channel_count),
    then the output will be of shape (instance_count, height * width * channel_count).

    :param name: The name of the layer.
    :type name: str

    """

    def __init__(self, name: str):
        # Super
        super().__init__(name=name)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """This method executes the transformation of this layer. 
        
        :param x: A tensor of shape (instance count, ...).
        :type x: numpy.ndarray

        :returns y (numpy.ndarray): The output of the transformation. Shape == (instance count, total_dimension_count).
        
        """
        
        return np.reshape(x, [x.shape[0], np.prod(x.shape[1:])])

class Dense(Layer):
    """This class defines a dense layer. A dense layer simply transforms a vector using a linear (matrix) operation and a bias. This is also known as affine transformation. 
    A subsequent non-linear activation function can be applied to increase the complexity of this transformation.

    :param name: The name of the layer.
    :type name: str

    :param input_dimensionality: The number of dimensions that the input vectors have.
    :type input_dimensionality: int

    :param output_dimensionality: The number of dimensions that the output vectors shall have.
    :type output_dimensionality: int

    :param activation_function: A string naming the activation function that shall be applied after the affine transformation. Any string from the set of keys in the 
    'layers.activation_functions' dictionary is legitimate.
    :type activation_function: str
    """

    def __init__(self, name: str, input_dimensionality: int, output_dimensionality: int, activation_function: str = 'none'):
        # Super
        super().__init__(name=name)
        
        # Copy arguments
        self.input_dimensionality = input_dimensionality
        self.output_dimensionality = output_dimensionality
        self.activation_function = activation_functions[activation_function]
        
        # Initialize weights
        self.weights['kernel:0'] = np.random.uniform(low=-1.0, high=1.0, size=[input_dimensionality, output_dimensionality])
        self.weights['bias:0'] = np.random.uniform(low=-1.0, high=1.0, size=[output_dimensionality])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """This method executes the transformation of this layer. 
        
        :param x: A tensor of shape (instance count, input_dimensionality), where input_dimensionality needs to be the same as specified in `layers.Dense.__init__()`.
        :type x: numpy.ndarray

        :returns y (numpy.ndarray): The output of the transformation. Shape == (instance count, output_dimensionality), where output_dimensionality is the same as specified in `layers.Dense.__init__()`.
        
        """

        return self.activation_function(np.dot(x, self.weights['kernel:0']) + self.weights['bias:0'])
    
    def load_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """
        Loads the weights for this layer.

        :param weights: A dictionary with keys 'kernel:0' and 'bias:0'. The value for 'kernel:0' shall be a numpy array of shape (input_dimensionality, output_dimensionality). The value for 'bias:0' shall be of shape (output_dimensionality). All of these dimensionalities are specified in `layers.Dense.__init__()`.
        :type weights: Dict[str, numpy.ndarray]
        """
        super().load_weights(weights=weights)

class Convolution2D(Layer):
    """This class defines a 2-dimensional convolution layer. It transforms a 2d image (that may have multiple channels, e.g. for color) to a set of k feature maps. 
    This happens by convolving the image with a set of k kernels. This means that each kernel scans the image and we compute the elementwise product of the current 
    kernel's matrix and the current patch of the image. The sum over these elementwise products plus a bias is then used to indicate the presence of the filtered 
    pattern in the current patch. This is repeated for all patches of the image, hence producing the current kernel's feature map. Importantly, although the 
    convolution is called 2d, the image patch as well as the kernel are actually 3d because we allow for multiple channels (e.g. color channels). A subsequent 
    non-linear activation function can be applied to increase the complexity of this transformation.

    :param name: The name of the layer.
    :type name: str

    :param input_channel_count: The number of channels that the input images will have, e.g. 3 for an rgb image.
    :type input_channel_count: int

    :param output_channel_count: The number of kernel maps that shall be created, i.e. k in the above explanation.
    :type output_channel_count: int

    :param kernel_size: The width and height of kernels. This is typically a small number, e.g. 4 if you want a 4*4 kernel.
    :type kernel_size: int

    :param stride: The step size used when iterating the input image. A stride of 1 means that patches of the input image are created with one vertical and/ or one 
    horizontal pixel between them. A stride of 2 means 2 such pixels between patches, etc..
    :type stride: int

    :param padding: For padding > 0, a border of zeros will be created around the input image. This border has width = padding.
    :type padding: int

    :param activation_function: A string naming the activation function that shall be applied after the affine transformation. Any string from the set of keys in the 
    'layers.activation_functions' dictionary is legitimate.
    :type activation_function: str
    """

    def __init__(self, name: str, input_channel_count: int, output_channel_count: int, kernel_size: int , stride: int = 1, padding: int = 0, activation_function: str= 'none') -> None:
        # Super        
        super().__init__(name=name)
        
        # Copy arguments
        self.input_channel_count = input_channel_count
        self.output_channel_count = output_channel_count
        self.kernel_size = kernel_size; self.stride = stride; self.padding = padding
        self.activation_function = activation_functions[activation_function]
        
        # Initialize weights
        self.weights['kernel:0'] = np.random.uniform(low=-1.0,high=1.0,size=[kernel_size, kernel_size, input_channel_count, output_channel_count])
        self.weights['bias:0'] = np.random.uniform(low=-1.0,high=1.0,size=[output_channel_count])
            
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """This method executes the transformation of this layer. 
        
        :param x: A tensor of shape (instance count, image height, image width, input_channel_count), where input_channel_count needs to be the same as specified in `layers.Convolution2D.__init__()`.
        :type x: numpy.ndarray

        :returns y (numpy.ndarray): The output of the transformation. Shape == (instance count, new width, new height, output_channel_count), where output_channel_count is the same as specified in `layers.Convolution2D.__init__()`. The new height and width depend on padding, stride, kernel size and original image shape.
        
        """

        # Shapes
        instance_count, _, _, _ = x.shape

        # Prepare weights
        kernel = np.repeat(self.weights['kernel:0'][np.newaxis,:,:,:,:], repeats=instance_count, axis=0)
        bias = np.repeat(self.weights['bias:0'][np.newaxis,:], repeats=instance_count, axis=0)
        
        # Iterate images to convolve
        y_hat = self.activation_function(Convolution2D.__iterate__(x=x, patch_size=self.kernel_size, stride=self.stride, padding=self.padding, output_channel_count=self.output_channel_count, function=lambda x_ij: np.sum(np.sum(np.sum(np.repeat(x_ij[:,:,:,:,np.newaxis], repeats=self.output_channel_count, axis=-1)*kernel, axis=1), axis=1), axis=1) + bias))
    
        # Outputs
        return y_hat

    def __iterate__(x: np.ndarray, patch_size: int, stride: int, padding: int, output_channel_count: int, function: callable) -> np.ndarray:
        """
        This method performs the scanning of the input image used e.g. in convolution.
            
        :param patch_size: The width and height of patches. This is typically a small number, e.g. 4 if you want a 4*4 patch.
        :type patch_size: int

        :param stride: The step size used when iterating the input image. A stride of 1 means that patches of the input image are created with one vertical and/or one 
        horizontal pixel between them. A stride of 2 means 2 such pixels between patches, etc..
        :type stride: int

        :param padding: For padding > 0, a border of zeros will be created around the input image. This border has width = padding.
        :type padding: int
        
        :param output_channel_count: The number of output dimensions that will be created by `function`.
        :type output_channel_count: int

        :param function: The function that shall be applied to each patch. It shall map an input of shape (instance_count, patch_size, patch_size, input_channel_count) to an output of shape (instance_count, output_channel_count). In convolution, this would be an elementwise multiplication of the current image patch with a weight tensor and an addition of a bias.
        :type function: Callable
        """
        

        ########## YOUR CODE STARTS HERE

        # Get input image shape
        instance_count, img_height, img_width, input_channel_count = x.shape

        # Apply padding (if any)
        if padding > 0:
            x = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant', constant_values=0)

        # Calculate output dimensions
        output_height = (img_height + 2 * padding - patch_size) // stride + 1
        output_width = (img_width + 2 * padding - patch_size) // stride + 1

        # Initialize output
        y = np.zeros((instance_count, output_height, output_width, output_channel_count))

        # Iterate over height and width to extract patches and apply the function
        for i in range(output_height):
            for j in range(output_width):
                # Ensure patch boundaries are valid
                start_i = i * stride
                start_j = j * stride
                end_i = start_i + patch_size
                end_j = start_j + patch_size

                if end_i <= img_height + 2 * padding and end_j <= img_width + 2 * padding:
                    # Extract patch from the input image
                    patch = x[:, start_i:end_i, start_j:end_j, :]
                    # Apply the provided function to the patch
                    y[:, i, j, :] = function(patch)

        return y
        
        ########## YOUR CODE ENDS HERE
        
    def load_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """
        Loads the weights for this layer.

        :param weights: A dictionary with keys 'kernel:0' and 'bias:0'. The value for 'kernel:0' shall be a numpy array of shape (kernel_size, kernel_size, input_channel_count, output_channel_count). The value for 'bias:0' shall be of shape (output_dimensionality). All of these dimensionalities are specified in `layers.Convolution2D.__init__()`.
        :type weights: Dict[str, numpy.ndarray]
        """
        super().load_weights(weights=weights)

class MaxPooling2D(Layer):
    """This class defines 2-dimensional maximum pooling operation. A MaxPooling2D layer is typically inserted after a Convolution2D layer to reduce the number of pixels in an image.
    It is similar to convolution in that it iterates an image. Yet, given a current image patch, it computes the maximum value across the patch. This happens separately for each channel. 
    A subsequent non-linear activation function can be applied to increase the complexity of this transformation.

    :param name: The name of the layer.
    :type name: str

    :param pooling_size: The width and height of the pools, i.e. image patches. This is typically a small number, e.g. 4 if you want 4*4 pools.
    :type pooling_size: int

    :param stride: The step size used when iterating the input image. A stride of 1 means that patches of the input image are created with one vertical and/ or one 
    horizontal pixel between them. A stride of 2 means 2 such pixels between patches, etc..
    :type stride: int

    :param padding: For padding > 0, a border of zeros will be created around the input image. This border has width = padding.
    :type padding: int

    :param activation_function: A string naming the activation function that shall be applied after the maximum pooling. Any string from the set of keys in the 
    'layers.activation_functions' dictionary is legitimate.
    :type activation_function: str """
    
    def __init__(self, name: str, pooling_size: int, stride: int=1, padding: int=0, activation_function: str='none'):
        # Super
        super().__init__(name=name)
        
        # Copy attributes
        self.pooling_size = pooling_size
        self.stride = stride
        self.padding = padding
        self.activation_function = activation_functions[activation_function]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """This method executes the transformation of this layer. 
        
        :param x: A tensor of shape (instance count, image height, image width, input_channel_count).
        :type x: numpy.ndarray

        :returns y (numpy.ndarray): The output of the transformation. Shape == (instance count, new width, new height, input_channel_count). The new height and width depend on padding, stride, pooling size and original image shape.
        
        """
        
        # Shapes
        _,_,_, output_channel_count = x.shape
        
        ########## YOUR CODE STARTS HERE

        # Compute y_hat
        # Compute y_hat using the Convolution2D.__iterate__ method
        y_hat = Convolution2D.__iterate__(
            x=x,
            patch_size=self.pooling_size,
            stride=self.stride,
            padding=self.padding,
            output_channel_count=x.shape[-1],  # Number of channels remains the same in max pooling
            function=lambda patch: np.max(patch, axis=(1, 2))  # Apply max pooling over each patch
        )

        ########## YOUR CODE ENDS HERE
        
        # Output
        return y_hat

class GatedRecurrentUnit(Layer):
    """
    The gated recurrent unit is suited to process sequences of inputs. While iterating a sequence, e.g. pen stroke coordinates, it 
    maintains a memory vector that keeps track of patterns that emerge over time. The sequence of the memory vector can be output,
    which is useful for stacking mutliple gated recurrent units on top of each other. Alternatively, one can choose to only return 
    the final memory vector which is useful e.g. for classification.

    :param name: The name of the layer.
    :type name: str

    :param input_dimensionality: The number of dimensions that each input time frame (vector) has.
    :type input_dimensionality: int

    :param input_dimensionality: The number of dimensions that the memory vector shall have.
    :type input_dimensionality: int

    :param return_state_sequence: Indicates whether the whole sequence of the memory vector shall be returned or just the final one.
    :type return_state_sequence: bool    
    """

    def __init__(self, name: str, input_dimensionality: int, output_dimensionality: int, return_state_sequence: bool = False):
        # Super
        super().__init__(name)

        # Attributes
        self.input_dimensionality = input_dimensionality
        self.output_dimensionality = output_dimensionality
        self.return_state_sequence = return_state_sequence


    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        This method executes the transformation of this layer. 
        
        :param x: A tensor of shape (instance count, time frame count, input_dimensionality).
        :type x: numpy.ndarray

        :returns y (numpy.ndarray): The output of the transformation. If return_state_sequence was set to True, then y has shape [instance count, time_frame_count, output_dimensionality], else [instance count, output_dimensionality].
        """

        # Shapes
        instance_count, time_step_count, input_dimensionality = x.shape

        # Prepend x with zeros to simplify indexing in the loop (see below)
        x = np.concatenate([np.zeros([instance_count, 1, input_dimensionality], dtype=x.dtype), x], axis=1)

        # Extract weights
        Wxz = self.weights['kernel:0'][:,:self.output_dimensionality] # Shape == (input_dimensionality, output_dimensionality)
        Wxr = self.weights['kernel:0'][:,self.output_dimensionality : 2*self.output_dimensionality] # Shape == (input_dimensionality, output_dimensionality)
        Wxh = self.weights['kernel:0'][:, 2*self.output_dimensionality:] # Shape == (input_dimensionality, output_dimensionality)

        Whz = self.weights['kernel:1'][:, : self.output_dimensionality] # Shape == (output_dimensionality, output_dimensionality)
        Whr = self.weights['kernel:1'][:, self.output_dimensionality : 2 * self.output_dimensionality] # Shape == (output_dimensionality, output_dimensionality)
        Whh = self.weights['kernel:1'][:, 2 * self.output_dimensionality:] # Shape == (output_dimensionality, output_dimensionality)

        # Iterate time steps, update h vector
        h = np.zeros((instance_count, 1+time_step_count, self.output_dimensionality), dtype=np.float32)
        for t in range(1, time_step_count+1):
            xt = x[:,t,:] # Shape == [instance_count, input_dimensionality]
            ht_minus_1 = h[:,t-1,:] # Shape == [instance_count, output_dimensionality].
            
            ########## YOU CODE STARTS HERE

            # Update Gate (zₜ)
            z_t = activation_functions["sigmoid"](
                np.dot(xt, Wxz) + np.dot(ht_minus_1, Whz)
            )

            # Reset Gate (rₜ)
            r_t = activation_functions["sigmoid"](
                np.dot(xt, Wxr) + np.dot(ht_minus_1, Whr)
            )

            # Candidate Hidden State (ĥₜ)
            h_tilde_t = activation_functions["tanh"](
                np.dot(xt, Wxh) + np.dot(r_t * ht_minus_1, Whh)
            )

            # Final Hidden State (hₜ)
            h[:, t, :] = z_t * ht_minus_1 + (1 - z_t) * h_tilde_t
            
            ########## YOUR CODE ENDS HERE

        if self.return_state_sequence: y = h[:,1:,:]
        else: y = h[:,-1,:]

        # Output
        return y

class Model():
    """
    This class can be used to build models that use a sequence of layers.

    :param layers: A list of layers through which inputs shall be passed to obtain predictions. The layers have to be compatibel with each other, meaning the shape of the output of one layer needs to match with the expected shape of the input of the next layer.
    :type layers: List[layers.Layer]
    """

    def __init__(self, layers: List[Layer]):
        self.layers = layers
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        This method executes the model's transformation by feeding `x` through the layers of `self`.

        :param x: The input. Its shape has to match the required shape of the first layer that was passed to `layers.Model.__init__()`.
        :type x: numpy.ndarray

        :returns y (numpy.ndarray): The output. Its shape corresponds to the output of the last layer that was passed to `layers.Model.__init__()`.
        """
        for layer in self.layers:
            x = layer(x)

        return x 
    
    def load_weights(self, weights: Dict[str, Dict[str, np.ndarray]]) -> None:
        """
        This method loads the weights for all layers of `self`. 

        :param weights: A dictionary whose keys are the names of layers of this model. The corresponding values are dictionaries. Each such dictionary shall store the weights in the format that the corresponding layer expects.
        :type weights: Dict[str, Dict[str, numpy.ndarray]]
        """

        for layer in self.layers:
            if layer.name in weights.keys(): layer.load_weights(weights[layer.name])

    def print_summary(self, x):
        print("\nModel summary for input of shape ", x.shape)
        table = [['Layer Index', 'Layer Class', 'Layer Name', 'Output shape', 'Weight Count']]
        
            
        for l, layer in enumerate(self.layers):
            x = layer(x)
            table.append([l, (str)(type(layer))[15:-2], layer.name, x.shape, layer.weight_count()])
        
        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
            
