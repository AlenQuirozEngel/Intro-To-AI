import numpy as np
import matplotlib.pyplot as plt
import layers as ls
from matplotlib.widgets import Button
from pylab import Rectangle
import pickle as pkl
import os

def plot_convolution_2D(x: np.ndarray) -> None:
    """
    This function plots the convolution of layers.Convolutioin2D for a single image and a single kernel. Note that plots scale all pixel-values to the range [0,1].
    """
    
    # Shapes
    instance_count, height, width, input_channel_count = x.shape

    # Setup kernels
    kernels = [
        np.array([[0,0,0,-1,1,1],
                  [0,0,-1,1,1,1],
                  [0,-1,1,1,1,-1],
                  [-1,1,1,1,-1,0],
                  [1,1,1,-1,0,0],
                  [1,1,-1,0,0,0]])[:,:,np.newaxis,np.newaxis], 
                  
        np.array([[1,1,-1,0,0,0],
                  [1,1,1,-1,0,0],
                  [-1,1,1,1,-1,0],
                  [0,-1,1,1,1,-1],
                  [0,0,-1,1,1,1],
                  [0,0,0,-1,1,1]])[:,:,np.newaxis,np.newaxis], 
                  
        np.array([[0,0,0,0,0,0],
                  [0,-1,-1,-1,-1,0],
                  [1,1,1,1,1,1],
                  [1,1,1,1,1,1],
                  [0,-1,-1,-1,-1,0],
                  [0,0,0,0,0,0]])[:,:,np.newaxis,np.newaxis], 
                  
        np.array([[0,0,1,1,0,0],
                  [0,-1,1,1,-1,0],
                  [0,-1,1,1,-1,0],
                  [0,-1,1,1,-1,0],
                  [0,-1,1,1,-1,0],
                  [0,0,1,1,0,0]])[:,:,np.newaxis,np.newaxis]]
    
    kernel = np.concatenate(kernels, axis=-1)
    bias = np.array([-3,-3,-3,-3])
    kernel_size, _, input_channel_count, output_channel_count = kernel.shape

    # Create layers
    conv = ls.Convolution2D(name='conv2d', input_channel_count=input_channel_count, output_channel_count=output_channel_count, kernel_size=kernel_size, stride=1, padding=0, activation_function='relu')
    
    # Ensure weights and biases are set in a reproducible manner
    conv.load_weights({'kernel:0': kernel, # Shape == [kernel_size, kernel_size, input_channel_count, output_channel_count]
                       'bias:0':bias}) # Shape == [output_channel_count]
    
    # Obtain output
    y_hat = conv(x)

    # Plot
    figure = plt.figure(figsize=(8,3))
    
    # Indices
    image_index = 0
    kernel_index = 0
  
    # Draw
    def draw_plots():
        
        # Update input Image
        gs = figure.add_gridspec(output_channel_count,3,width_ratios=[1,1/output_channel_count,1])
        axes = figure.add_subplot(gs[:,0]); plt.title("Input")
        axes.imshow(x[image_index])
        axes.set_axis_off()

        # Update kernel displays
        for f in range(output_channel_count):
            axes = figure.add_subplot(gs[f,1])
            
            axes.clear()
            axes.imshow(conv.weights['kernel:0'][:,:,:,f]/np.max(conv.weights['kernel:0'])); axes.set_ylabel(f)
            axes.set_xticks([]); axes.set_yticks([])

            if f == kernel_index: 
                rectangle = Rectangle(xy=(-0.5,-0.5), width=kernel_size, height=kernel_size,fill=False,linewidth=5,color='red')
                axes.add_patch(rectangle)

        # Update output display
        axes = figure.add_subplot(gs[:,2]); plt.title("Output")
        axes.imshow(y_hat[image_index,:,:,kernel_index]/np.max(y_hat))
        axes.set_axis_off()
        
        # Show
        plt.show()
    
    # Button handlers
    def next_image_click(event):
        nonlocal image_index
        image_index = (image_index + 1) % instance_count
        draw_plots()

    def next_kernel_click(event):
        nonlocal kernel_index
        kernel_index = (kernel_index + 1) % output_channel_count
        draw_plots()

    # Buttons
    image_button_position = plt.axes([0.22, 0.01, 0.12, 0.07]) # x,y, width, height
    image_button = Button(image_button_position, 'Next Image', color='white', hovercolor='gray')
    image_button.on_clicked(next_image_click)

    kernel_button_position = plt.axes([0.455, 0.01, 0.12, 0.07]) # x,y, width, height
    kernel_button = Button(kernel_button_position, 'Next Kernel', color='white', hovercolor='gray')
    kernel_button.on_clicked(next_kernel_click)

    # First draw call
    draw_plots()

def plot_classification(x: np.ndarray, y_hat: np.ndarray, scatter=False) -> None:
        
    # Shapes
    instance_count, output_channel_count = y_hat.shape

    # Plot
    figure = plt.figure(figsize=(4,3))
    
    # Indices
    image_index = 0
  
    # Draw
    plots = []
    def draw_plots():
        # Clear old plots
        nonlocal plots
        if len(plots) > 0: 
            for plot in plots: plot.remove()
            plots=[]
        # Update input Image
        gs = figure.add_gridspec(1,2,width_ratios=[5,1])
        axes = figure.add_subplot(gs[:,0]); plt.title("Input")
        if scatter: 
            points = x[image_index,x[image_index,:,0]>0,:]
            plots.append(axes.scatter(points[:,0], points[:,1]))
            for p, point in enumerate(points):
                plots.append(plt.text(point[0]+1, point[1]+1, (str)(p)))
            plt.xlim(1,28); plt.ylim(1,28)
        else: axes.imshow(x[image_index])
        axes.set_axis_off()

        # Update output display
        axes = figure.add_subplot(gs[:,1]); plt.title("Output")
        axes.imshow(y_hat[image_index,np.newaxis].T)
        axes.set_xticks([]); axes.set_yticks(range(10))
        
        # Show
        plt.show()
    
    # Button handlers
    def next_image_click(event):
        nonlocal image_index
        image_index = (image_index + 1) % instance_count
        draw_plots()

    # Buttons
    image_button_position = plt.axes([0.3, 0.01, 0.3, 0.07]) # x,y, width, height
    image_button = Button(image_button_position, 'Next Image', color='white', hovercolor='gray')
    image_button.on_clicked(next_image_click)

    # First draw call
    draw_plots()

if __name__ == "__main__":

    # Load image data
    with open('x_test.npy', 'rb') as f:
        x = np.load(f)[:25,:] # Only take the first few instances for performance reasons

    # THIS PART PLOTS A DENSE NEURAL NETWORK FOR HANDWRITTEN DIGIT CLASSIFICATION. Set if True/ if False depending on whether you want to execute it
    #############################################################################################################
    if True:
        # Turn images to vectors
        instance_count, height, width, channel_count = x.shape
        
        # Build default model
        layers = [ls.Flatten(name='flat'),
                  ls.Dense(name='dense_1', input_dimensionality=height*width*channel_count, output_dimensionality=128, activation_function='relu'),
                  ls.Dense(name='dense_2', input_dimensionality=128, output_dimensionality=10, activation_function='softmax')]

        model = ls.Model(layers=layers)
        model.print_summary(x=x)

        # Load parameters
        with open('DNN weights.pkl', 'rb') as f:
            model.load_weights(weights=pkl.load(f))

        # Predict
        y_hat = model(x=x)

        # Plot
        plot_classification(x=x, y_hat=y_hat)

    
    # THIS PART PLOTS THE CONVOLUTION OPERATION OF A SINGLE LAYER FOR HANDWRITTEN DIGIT CLASSIFICATION. Set if True/ if False depending on whether you want to execute it
    ##############################################################################################################
    if False:
        # Plot
        plot_convolution_2D(x=x[:10,:])


    # THIS PART BUILDS A CONVOLUTIONAL NEURAL NETWORK FOR HANDWRITTEN DIGIT CLASSIFICATION. Set if True/ if False depending on whether you want to execute it
    ##############################################################################################################
    if True:
        # Build model
        layers = [

             # Convolution layer with 32 output channels
            ls.Convolution2D(name='conv2d', input_channel_count=1, output_channel_count=32, kernel_size=3, stride=1, padding=1, activation_function='relu'),
            
            # Max Pooling layer with pooling size of 2
            ls.MaxPooling2D(name='pool2d', pooling_size=2, stride=2, padding=0),

            # Second Convolution layer with 64 output channels
            ls.Convolution2D(name='conv2d_1', input_channel_count=32, output_channel_count=64, kernel_size=3, stride=1, padding=0, activation_function='relu'),
            
            # Second Max Pooling layer with pooling size of 2
            ls.MaxPooling2D(name='pool2d_1', pooling_size=2, stride=2, padding=0),

            # Flatten layer to reshape the tensor
            ls.Flatten(name='flat'),

            # Fully connected (Dense) layer for classification with softmax activation
            ls.Dense(name='dense', input_dimensionality=6*6*64, output_dimensionality=10, activation_function='softmax')
            
        ]

        model = ls.Model(layers=layers)
        model.print_summary(x=x)

        # Load parameters
        with open('CNN weights.pkl', 'rb') as f:
            model.load_weights(weights=pkl.load(f))

        # Predict
        y_hat = model(x=x)

        # Plot
        plot_classification(x=x, y_hat=y_hat)


    # THIS PART BUILDS A RECURRENT NEURAL NETWORK FOR HANDWRITTEN DIGIT CLASSIFICATION. Set if True/ if False depending on whether you want to execute it
    ##############################################################################################################
    if True:
        # Build model
        layers = [
        
        # Gated Recurrent Unit layer
        ls.GatedRecurrentUnit(name='gru', input_dimensionality=2, output_dimensionality=32, return_state_sequence=False),

        # Fully connected (Dense) layer for classification
        ls.Dense(name='dense', input_dimensionality=32, output_dimensionality=10, activation_function='softmax')
            
        ]

        model = ls.Model(layers=layers)
        

        # Load parameters
        with open('RNN weights.pkl', 'rb') as f:
            model.load_weights(weights=pkl.load(f))

        with open('x_test_RNN.pkl', 'rb') as f:
            test_data = pkl.load(f)

        # Pad sequences
        number_of_instances = 25
        max_length = 0
        for i in range(number_of_instances):
            sequence_length, coordinate_count = test_data[i].shape
            if max_length < sequence_length: max_length = sequence_length

        x = np.zeros([number_of_instances, max_length, 2], dtype=np.float32)
        for i in range(number_of_instances):
            sequence_length, coordinate_count = test_data[i].shape
            x[i, -sequence_length:, :] = test_data[i]

        # Model summary
        model.print_summary(x=x)
        
        # Predict
        y_hat = model(x=x)

        # Plot
        plot_classification(x=x, y_hat=y_hat, scatter=True)