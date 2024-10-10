import layers as ls
import numpy as np
import unittest
import matplotlib.pyplot as plt
import os
import pickle as pkl

class FlattenLayerTests(unittest.TestCase):

    def test_call(self):
        # Example input
        x = np.array([
            [
                [[1,2,3], [4,5,6], [7,8,9]],
                [[10,11,12],[13,14,15], [16,17,18]],
                [[19,20,21],[22,23,24],[25,26,27]]
            ],
            [
                [[11,12,13], [14,15,16], [17,18,19]],
                [[110,111,112],[113,114,115], [116,117,118]],
                [[119,120,121],[122,123,124],[125,126,127]]
            ]
        ])

        # Corresponding target
        y = np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],
                      [11,12,13,14,15,16,17,18,19,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127]])
        
        # Obtain output
        flatten = ls.Flatten(name='flat')
        y_hat = flatten(x=x)

        # Evaluate
        self.assertEqual(type(y), type(y_hat))
        self.assertEqual(y.shape, y_hat.shape)
        self.assertTrue((y == y_hat).all())

class DenseLayerTests(unittest.TestCase):

    def test_call_case_1(self):
        """Test case 1 tests whether a standard dense layer with 784 input dimensions and 10 output dimensions works as intended."""
        
        # Create layer
        dense = ls.Dense(name='dense', input_dimensionality=784, output_dimensionality=10, activation_function='none')
        
        # Ensure weights and biases are set in a reproducible manner
        with open(os.path.join('unit test data', 'dense', 'test case 1 weights.pkl'), 'rb') as f:
            weights = pkl.load(f)
            dense.load_weights(weights={'kernel:0': weights['dense/kernel:0'], 'bias:0':weights['dense/bias:0']})
        
        # Create input
        with open('x_test.npy', 'rb') as f:
            x = np.reshape(np.load(f), [10000, 784])

        # Create target output
        with open(os.path.join('unit test data', 'dense', 'test case 1 y_test.npy'), 'rb') as f:
            y = np.load(f)

        # Obtain output
        y_hat = dense(x)
        
        # Evaluate results
        self.assertEqual(type(y), type(y_hat))
        self.assertEqual(y.shape, y_hat.shape)
        self.assertTrue(np.allclose(y, y_hat, atol=1e-6))

class ConvolutionLayerTests(unittest.TestCase):

    def test_call_case_1(self):
        """Test case 1 tests whether a standard convolution with a 1 channel input and a 4 channel output and a filter-size of 3x3 works. 
        Stride=1, padding=0 and activation_function='none' are kept at their default values and are thus not tested with this unit test."""
        
        # Create layer
        conv = ls.Convolution2D(name='conv2d', input_channel_count=1, output_channel_count=4, kernel_size=3, stride=1, padding=0)
        
        # Ensure weights and biases are set in a reproducible manner
        with open(os.path.join('unit test data', 'convolution', 'test case 1 weights.pkl'), 'rb') as f:
            weights = pkl.load(f)
            conv.load_weights(weights={'kernel:0': weights['conv2d/kernel:0'], 'bias:0':weights['conv2d/bias:0']})
        
        
        # Create input
        with open('x_test.npy', 'rb') as f:
            x = np.load(f)[:5,:] # Use only the first few instances for performance

        # Create target output
        with open(os.path.join('unit test data', 'convolution', 'test case 1 y_test.npy'), 'rb') as f:
            y = np.load(f)[:5,:] # Use only the first few instances for performance

        # Obtain output
        y_hat = conv(x)
        
        # Evaluate results
        self.assertEqual(type(y), type(y_hat))
        self.assertEqual(y.shape, y_hat.shape)
        self.assertTrue(np.allclose(y, y_hat, atol=1e-6))

    def test_call_case_2(self):
        """Test case 1 tests whether a standard convolution with a 1 channel input and a 4 channel output, a filter-size of 3x3 and a stride of 2 works. 
        Padding=0 and activation_function='none' are kept at their default values and are thus not tested with this unit test."""
        
        # Create layer
        conv = ls.Convolution2D(name='conv2d', input_channel_count=1, output_channel_count=4, kernel_size=3, stride=2, padding=0)
        
        # Ensure weights and biases are set in a reproducible manner
        with open(os.path.join('unit test data', 'convolution', 'test case 2 weights.pkl'), 'rb') as f:
            weights = pkl.load(f)
            conv.load_weights(weights={'kernel:0': weights['conv2d/kernel:0'], 'bias:0':weights['conv2d/bias:0']})
        
        
        # Create input
        with open('x_test.npy', 'rb') as f:
            x = np.load(f)[:5,:] # Use only the first few instances for performance

        # Create target output
        with open(os.path.join('unit test data', 'convolution', 'test case 2 y_test.npy'), 'rb') as f:
            y = np.load(f)[:5,:] # Use only the first few instances for performance

        # Obtain output
        y_hat = conv(x)
        
        # Evaluate results
        self.assertEqual(type(y), type(y_hat))
        self.assertEqual(y.shape, y_hat.shape)
        self.assertTrue(np.allclose(y, y_hat, atol=1e-6))

    def test_call_case_3(self):
        """Test case 1 tests whether a standard convolution with a 1 channel input and a 4 channel output, a filter-size of 5x5 and a padding of 2 works. 
        Stride=1 and activation_function='none' are kept at their default values and are thus not tested with this unit test."""
        
        # Create layer
        conv = ls.Convolution2D(name='conv2d', input_channel_count=1, output_channel_count=4, kernel_size=5, stride=1, padding=2)
        
        # Ensure weights and biases are set in a reproducible manner
        with open(os.path.join('unit test data', 'convolution', 'test case 3 weights.pkl'), 'rb') as f:
            weights = pkl.load(f)
            conv.load_weights(weights={'kernel:0': weights['conv2d/kernel:0'], 'bias:0':weights['conv2d/bias:0']})
        
        # Create input
        with open('x_test.npy', 'rb') as f:
            x = np.load(f)[:5,:] # Use only the first few instances for performance

        # Create target output
        with open(os.path.join('unit test data', 'convolution', 'test case 3 y_test.npy'), 'rb') as f:
            y = np.load(f)[:5,:] # Use only the first few instances for performance

        # Obtain output
        y_hat = conv(x)
        
        # Evaluate results
        self.assertEqual(type(y), type(y_hat))
        self.assertEqual(y.shape, y_hat.shape)
        self.assertTrue(np.allclose(y, y_hat, atol=1e-6))

class MaxPoolingLayerTests(unittest.TestCase):

    def test_call(self):
        # Create layer
        pool = ls.MaxPooling2D(name='pool', pooling_size=4, stride=1, padding=0)
        
        # Create input
        x = np.repeat(np.reshape(np.array([[0,0,0,1,0.1,0,0,0,0,0],
                                           [0,0.5,0,0,2.1,1,0,0,0,0],
                                           [0,1,0,0,0,1,3.1,0,0,0],
                                           [0,-1,0,0,0,0,-1,1,0,0],
                                           [0,0.25,0,0,0,1,0.1,0,0,0],
                                           [0,0,0,0,1,-21,0,0,0,0]]), [1,6,10,1]),repeats=3,axis=3)

        # Create target output
        y = np.array([[[[1.,  1.,  1., ],
                        [2.1, 2.1, 2.1],
                        [2.1, 2.1, 2.1],
                        [3.1, 3.1, 3.1],
                        [3.1, 3.1, 3.1],
                        [3.1, 3.1, 3.1],
                        [3.1, 3.1, 3.1]],

                        [[1.,  1.,  1. ],
                        [2.1, 2.1, 2.1],
                        [2.1, 2.1, 2.1],
                        [3.1, 3.1, 3.1],
                        [3.1, 3.1, 3.1],
                        [3.1, 3.1, 3.1],
                        [3.1, 3.1, 3.1]],

                        [[1.,  1.,  1. ],
                        [1.,  1.,  1. ],
                        [1.,  1.,  1. ],
                        [3.1, 3.1, 3.1],
                        [3.1, 3.1, 3.1],
                        [3.1, 3.1, 3.1],
                        [3.1, 3.1, 3.1]]]]
                        )

        # Obtain output
        y_hat = pool(x)
        
        # Evaluate results
        self.assertEqual(type(y), type(y_hat))
        self.assertEqual(y.shape, y_hat.shape)
        self.assertTrue((y == y_hat).all())

class GatedRecurrentUnitLayerTests(unittest.TestCase):

    def test_call_case_1(self):
        """This test case tests whether a random requence of inputs can be transformed to a sequence of state vectors by the gated recurrent unit."""

        # Shapes
        time_steps = 10
        input_dimensionality = 9
        output_dimensionality = 5
        instance_count = 32
        
        # Load input
        with open(os.path.join('unit test data', 'gated recurrent unit', 'test case 1 x.npy'), 'rb') as f:
            x = np.load(f)

        # Create layer
        gru = ls.GatedRecurrentUnit(name='gru', input_dimensionality=input_dimensionality, output_dimensionality=output_dimensionality, return_state_sequence=True)
        
        # Ensure weights are set in a reproducible manner
        with open(os.path.join('unit test data', 'gated recurrent unit', 'test case 1 weights.pkl'), 'rb') as f:
            gru.load_weights(pkl.load(f))
        
        # Load target 
        with open(os.path.join('unit test data', 'gated recurrent unit', 'test case 1 y.npy'), 'rb') as f:
            y = np.load(f)
            
        # Obtain output
        y_hat = gru(x)
        
        # Evaluate results
        self.assertEqual(type(y), type(y_hat))
        self.assertEqual(y.shape, y_hat.shape)
        self.assertTrue(np.allclose(y, y_hat, atol=1e-5))

class ModelLayerTests(unittest.TestCase):

    def test_call_case_1(self):
        """Test case 1 tests whether a simple dense neural network with flatten and dense layer can be set up. This involves the load_weights and __call__ methods."""
        
        # Create layers
        layers = [
            ls.Flatten(name='flat'),
            ls.Dense(name='dense', input_dimensionality=28*28, output_dimensionality=32, activation_function='softmax')
        ]

        model = ls.Model(layers=layers)

        # Ensure weights and biases are set in a reproducible manner
        with open(os.path.join('unit test data', 'model', 'test case 1 weights.pkl'), 'rb') as f:
            weights = pkl.load(f)

        model.load_weights(weights=weights)
        
        # Create input
        with open('x_test.npy', 'rb') as f:
            x = np.load(f)[:5] # Only use the first few instances for speed

        # Create target output
        with open(os.path.join('unit test data', 'model', 'test case 1 y_test.npy'), 'rb') as f:
            y = np.load(f)[:5] # Only use the first few instances for speed

        # Obtain output
        y_hat = model(x)
        
        # Evaluate results
        self.assertEqual(y.shape, y_hat.shape)
        self.assertTrue(np.allclose(y, y_hat, atol=1e-6))
        

if __name__ == '__main__':
    unittest.main()
    