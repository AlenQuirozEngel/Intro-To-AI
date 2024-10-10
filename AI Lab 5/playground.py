
    def test_call_case_2(self):
        """Test case 2 tests whether a simple convolutional neural network for classification can be built here. The model has its layers
        set up as a convolution, maxpooling, convolution, maxpooling, flatten and dense. All configurations are set to their defaults,
        except for the activation functions which are relu, relu and softmax."""
        
        # Create layers
        layers = [
            ls.Convolution2D(name='conv2d', input_channel_count=1, output_channel_count=4, kernel_size=3, stride=1, padding=0, activation_function='relu'),
            ls.MaxPooling2D(name='pool', pooling_size=2, stride=1, padding=0),
            ls.Convolution2D(name='conv2d_1', input_channel_count=4, output_channel_count=64, kernel_size=3, stride=1, padding=0, activation_function='relu'),
            ls.MaxPooling2D(name='pool_2', pooling_size=2, stride=1, padding=0),
            ls.Flatten(name='flat'),
            ls.Dense(name='dense', input_dimensionality=30976, output_dimensionality=10, activation_function='softmax')
        ]

        model = ls.Model(layers=layers)

        # Ensure weights and biases are set in a reproducible manner
        with open(os.path.join('unit test data', 'model', 'test case 2 weights.pkl'), 'rb') as f:
            weights = pkl.load(f)
        
        model.load_weights(weights=weights)
        
        # Create input
        with open('x_test.npy', 'rb') as f:
            x = np.load(f)[:5] # Only use the first few instances for speed

        # Create target output
        with open(os.path.join('unit test data', 'model', 'test case 2 y_test.npy'), 'rb') as f:
            y = np.load(f)[:5] # Only use the first few instances for speed

        # Obtain output
        y_hat = model(x)
        
        # Evaluate results
        self.assertEqual(y.shape, y_hat.shape)
        self.assertTrue(np.allclose(y, y_hat, atol=1e-6))
