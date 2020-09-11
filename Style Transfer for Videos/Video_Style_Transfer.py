import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import glob
import numpy as np
from PIL import Image
import PIL
import keras
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications import imagenet_utils
from keras import backend as K
from scipy.ndimage.filters import gaussian_filter
from functools import partial
import pprint
import time
from imageio import imsave
from matplotlib.pyplot import imshow
from io import BytesIO
import matplotlib.pyplot as plt

ITERAT = 100
learning_rate = 10

if not os.path.exists('results'):
    os.makedirs('results')
np.warnings.filterwarnings('ignore')
def plot(img, scale=1, dpi=80):
    plt.figure(figsize=(img.shape[0]*scale/dpi, img.shape[1]*scale/dpi), dpi=dpi)
    imshow(img)
    
# Prepare a tensor to be displayed as image
def normalize(x):
    x = x.copy().astype(float)
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    
    x += 0.5
    x = np.clip(x, 0, 1)
    # convert to RGB array
    x *= 255
    return x.astype('uint8')

def show(img):
	temp = img.copy() / 256 

	# plt.imshow(temp)
	# plt.show()


def resize_array(array, size):
    image = PIL.Image.fromarray(array.astype('uint8').copy())
    image_resized = image.resize(size, PIL.Image.ANTIALIAS)
    return np.asarray(image_resized).astype(float)


# Output image size
image_size = (300, 300)




def random_image(size,):
    width, height = size
    img = np.random.random(size=(width, height, 3))
    img_array = img * 256
    return img_array


def load_image(size, filename=None):
    width, height = size
    
    img = PIL.Image.open(filename)
    img = img.resize((width, height))
    img_array = np.asarray(img.copy())
    return img_array


image = random_image(size=image_size)
show(image)



graph = tf.Graph()
config = tf.ConfigProto()

sess = tf.InteractiveSession(graph=graph, config=config)
K.set_session(sess)

K.set_learning_phase(0)

xx_num = int(60)

def setup_model(initial_value, model_name='Inception5h'):
    '''Load the model. Use a TF tensor as input just for fun.
    Args: 
        initial_value: The initial value of the input tensor. Mainly used for size.
        model_name: Whether to normalize the input image.
    Returns:
        Tuple of (model, # The loaded keras model
            input_tensor, # The tensor that feeds the model
            content_layers, # The content layers of this model as tensors
            style_layers, # The style layers of this model as tensors
            preprocess_func, # Preprocesses an image for the model
            deprocess_func # Returns preprocessed image back to normal.
    '''
    # Prepare tensor for input image
    image_tensor = tf.Variable(initial_value)

    if model_name == 'VGG16' or model_name == 'VGG19':
        # These two models share a lot, so define them together
        if model_name == 'VGG16':
            # VGG 16 model
            model = VGG16(include_top=False, weights='imagenet',
                          input_tensor=image_tensor)
        elif model_name == 'VGG19':
            model = VGG19(include_top=False, weights='imagenet',
                          input_tensor=image_tensor)

        # Preprocesses an image for the model
        def preprocess_func(x):
            x = x.copy().astype(float)
            rank = len(x.shape)
            if (rank == 3):
                # Add extra batch dimension
                x = np.expand_dims(x, axis=0)
            x[:, :, :, 2] -= 103.939
            x[:, :, :, 1] -= 116.779
            x[:, :, :, 0] -= 123.68

            # Flip the channels from RGB to BGR
            x = x[:, :, :, ::-1]
            return x

        # Returns preprocessed image back to normal.
        def deprocess_func(x):
            x = np.asarray(x).copy()
            rank = len(x.shape)
            if (rank == 4):
                # Remove extra batch dimension
                x = np.squeeze(x, axis=0)

            # Flip the channels from BGR to RGB
            x = x[:, :, ::-1]

            # Remove zero-center by mean pixel
            x[:, :, 2] += 103.939
            x[:, :, 1] += 116.779
            x[:, :, 0] += 123.68

            x = np.clip(x, 0, 255).astype('uint8')
            return x

        # Define the style layers
        style_layers = [model.get_layer('block1_conv1').output,
                        model.get_layer('block2_conv1').output,
                        model.get_layer('block3_conv1').output,
                        model.get_layer('block4_conv1').output,
                        model.get_layer('block5_conv1').output]

        # Define the content layers
        content_layers = model.get_layer('block4_conv2').output

    # TODO: Add other models

    return model, image_tensor, content_layers, style_layers, preprocess_func, deprocess_func

with graph.as_default():
    with graph.name_scope("model") as scope:
        # Use the image size (with a batch dimension) to feed the model input.
        # We do so that the input has size and the convolutional layers
        # will be sized as well.
        initial = np.expand_dims(image, axis=0).astype('float32')

        # Setup the model
        (model, input_tensor, content_layers, style_layers,
         preprocess, deprocess) = setup_model(
            initial_value=initial,
            model_name='VGG16')



# print(model, input_tensor, content_layers, style_layers, preprocess, deprocess)



# Summary of the model layers
# print(model.summary())


style_image = load_image(size=image_size, filename='style.jpg')
show(style_image)

content_image_lis = glob.glob('Extracted_Frames/*')
content_image_lis = sorted(content_image_lis)
print(content_image_lis)
# exit(0)
def style_loss(current, computed):
    '''Define the style loss between a tensor and an np array.
    Args:
        current: tf.Tensor. The style activations of the current image.
        computed: np array. The style activations of the style input image.
    '''
    style_losses = []
    for layer1, layer2 in zip(current, computed):
        _, height, width, number = map(lambda i: i, layer2.shape)
        size = height * width * number

        # Compute layer1 Gram matrix
        feats1 = tf.reshape(layer1, (-1, number))
        layer1_gram = tf.matmul(tf.transpose(feats1), feats1) / size
        # Compute layer2 Gram matrix
        feats2 = tf.reshape(layer2, (-1, number))
        layer2_gram = tf.matmul(tf.transpose(feats2), feats2) / size

        dim1, dim2 = map(lambda i: i.value, layer1_gram.get_shape())
        loss = tf.sqrt(tf.reduce_sum(
            tf.square((layer1_gram - layer2_gram) / (number * number))))
        style_losses.append(loss)
    return tf.add_n(style_losses)

def total_variation_loss(image, image_shape=None):
    if image_shape:
        width, height, channels = image_shape
    else:
        # Try to get the image size from the tensor
        dims = image.get_shape()
        width = dims[1].value
        height = dims[2].value
        channels = dims[3].value

    tv_x_size = width * (height - 1) * channels
    tv_y_size = (width - 1) * height * channels

    return (
        tf.reduce_sum(tf.abs(image[:, 1:, :, :] - image[:, :-1, :, :])) +
        tf.reduce_sum(tf.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    )

def content_loss(current, computed):
    # Currently only for a single layer
    _, height, width, number = computed.shape
    size = height * width * number
    return tf.sqrt(tf.nn.l2_loss(current - computed) / size)

def setup_gradient(input_tensor, result_tensor):
    '''Setup the gradient of the input tensor w.t.r 
    to the result tensor.
    Args: 
        input_tensor: The input features tensor.
        result_tensor: The tensor that we want to maximize.
    '''
    
    #get Mean of the tensor
    excitement_score = tf.reduce_mean(result_tensor)

    # Gradients give us how to change the input (input_tensor)
    # to increase the excitement_score.
    # We get the first result only since the model is designed to
    # work on batches, and we only use single image.
    gradient = tf.gradients(excitement_score, input_tensor)[0]

    # Normalize the gradient by its L2 norm.
    # Disabled for now.
    # gradient /= (tf.sqrt(tf.reduce_mean(tf.square(gradient)))
    #                            + 1e-5)

    return gradient, excitement_score
for file in content_image_lis:
    content_image = load_image(size=image_size, filename=file)
    # show(content_image)



    # Compute the activations
    style_layers_computed = sess.run(
        style_layers,
        feed_dict={input_tensor: preprocess(style_image)})

    content_layers_computed = sess.run(
        content_layers,
        feed_dict={input_tensor: preprocess(np.expand_dims(content_image, 0))})

    ###################################################
    ###################################################
    # Loss Functions
    ###################################################
    ###################################################



    # ## Step 4
    # Difference between content layers of result image and content layers of content image.




    # ## Step 5
    # Total variation loss, designed to keep the generated image locally coherent.
    # https://en.wikipedia.org/wiki/Total_variation_denoising
    # 



    # ## Step 6.
    # Define the gradient to change a result image
    # 

    # In[235]:



    # how to change the input to increase the excitement score
    # Is neuron ka fire badhana hai

    # How much content, style and total variance loss contribute to the
    # total loss.
    content_weight = 1e3
    style_weight = 1e6
    tv_weight = 1e-2  # 1e-3

    # Set up the style, content, total variation, as well as total loss
    # and use them to define the gradient.
    with graph.as_default():
        with graph.name_scope("style_loss") as scope:
            style_loss_op = style_weight *             style_loss(style_layers, style_layers_computed)
        with graph.name_scope("content_loss") as scope:
            content_loss_op = content_weight *             content_loss(content_layers, content_layers_computed)
        with graph.name_scope("tv_loss") as scope:
            tv_loss_op = tv_weight * total_variation_loss(input_tensor)
        with graph.name_scope("loss") as scope:
            total_loss_op = style_loss_op + content_loss_op + tv_loss_op
        with graph.name_scope("gradient") as scope:
            gradient_op, score_op = setup_gradient(input_tensor, total_loss_op)


    ###DONE

    # Uncomment this to display the whole TF Graph interactively
    # display_graph(graph.as_graph_def())


    # ## Step 7
    # Modify the result image by the gradient.

    # ### Define optimization procedure


    def get_uninitialized_variables(variables=None, session=None):
        """
        Get uninitialized variables in a session as a list.
            Args: 
                variables: list of tf.Variable. Get uninitiliazed vars 
                    from these. If none, gets all uinitialized vars in session.
                session: tf.Session to find uninitialized vars in. If none
                    uses default session.
            Returns:
                Uninitialized variables within `variables`.
                If `variables` not specified, return all uninitialized variables.
        """
        if not session:
            session = tf.get_default_session()
        if variables is None:
            variables = tf.global_variables()
        else:
            variables = list(variables)
        init_flag = session.run(
            tf.stack([tf.is_variable_initialized(v) for v in variables]))
        return [v for v, f in zip(variables, init_flag) if not f]


    def initialize_variables():
        '''Initialize the internal variables the optimizer uses.
            We could do tf.global_variables_initializer().eval() to 
            initialize all variables but this messes up the keras model.'''
        # Get uninitialized vars and their initializers
        uninitialized_vars = get_uninitialized_variables()
        initializers = [var.initializer for var in uninitialized_vars]

        # Print uninitialized variables
        print('Uninitialized variables:')
        print([initializer.name for initializer in initializers])

        # Initialize the variables
        _ = [initializer.run() for initializer in initializers]


    def init_input(show_image=False):
        ''' Define start image in the input tensor.
            The image we are going to start with is the content image, 
            with random petrubations (needed, otherwise TV loss converges to nan).
        Args:
            show_image: Whether to display the generated image.
        '''
        # Define random 0 to 1 image with size (batch_size, image_size, channels)
        #  mean=0.5, stddev=.5
        initial_random = tf.random_normal(
            mean=1, stddev=.01, shape=(1,) + image_size + (3,))

        # Use the content image
        initial_content = preprocess(np.expand_dims(content_image, axis=0))

        # Init the input tensor
        input_tensor.initializer.run()
        input_tensor.assign(
            tf.clip_by_value(initial_content * initial_random, 0, 255)
        ).eval()

        # Show the input tensor
        if show_image:
            show(np.squeeze(input_tensor.eval()))

    # Helper to print the losses.


    def print_progress(i,
                    iterations,
                    loss_computed,
                    style_loss_computed,
                    content_loss_computed,
                    tv_loss_computed):
        print('Iteration %d/%d Content L: %g Style L: %g TV L: %g Total L: %g' % (
            i,
            iterations,
            content_loss_computed,
            style_loss_computed,
            tv_loss_computed,
            loss_computed
        ))


    def run_optimization(train_step, iterations=ITERAT, print_n_times=10):
        ''' Run the optimization for a number of iterations. 
        Args:
            train_step: The op that gets executed every iteration.
            iterations: How many times to run the optimization. Recommended at
                least 500 - 1000 iterations for a good quality image. Good style
                should be visible even after 100 iters.
            print_n_times: Int, How many times to print the progress.
        Return:
            A list of losses during the optimization like so:
            [(time elapsed, loss)]
        '''
        if print_n_times == 0:
            # Dont print at all
            print_every_n = iterations + 1
        else:
            print_every_n = max(iterations // print_n_times, 1)

        # Keep only the image with the lowest loss
        # (in case we converge).
        best_loss = float('inf')
        best = None

        losses = []

        # To compute total optimization time
        start_time = time.time()

        # Optimization loop
        for i in range(iterations):
            # Keep the input_tensor between 0 and 255
            # (gives slightly better output, slows optimization by factor of 2)
    #         input_tensor.assign(tf.clip_by_value(input_tensor, 0, 255)).eval()

            # Run the training (train_step), and get the losses
            (_, result_image, loss_computed,
            style_loss_computed, content_loss_computed,
            tv_loss_computed) = sess.run(
                [train_step, input_tensor, score_op, style_loss_op, content_loss_op,
                tv_loss_op])

            wall_time = time.time() - start_time

            losses.append((wall_time, loss_computed))

            # Print progress
            if i % print_every_n == 0:
                print_progress(i, iterations, loss_computed,
                            style_loss_computed, content_loss_computed,
                            tv_loss_computed)

            # skip this for now.
            # if loss_computed < best_loss:
            #    best_loss = loss_computed
            #    best = result_image

        total_time = time.time() - start_time
        print ('Training took {:.0f} seconds or {:.2f} s/iteration !'.format(
            wall_time,
            wall_time / iterations))

        return losses


    def optimize(optimizer, iterations=ITERAT, new_image=True):
        ''' Define and run the optimization.
        Args:
            optimizer: The optimizer to use.
            iterations: Number of times to run optimizer.
            new_image: Whether to start with a new image, or continue with the 
                previous one.
        Return: A tuple: 
            A list of losses during the optimization like so: [(time elapsed, loss)]
            The image
        '''
        with graph.as_default():
            # Compute the gradients for a list of variables.
            grads_and_vars = optimizer.compute_gradients(score_op, [input_tensor])
            # Op that ask the optimizer to apply the gradients.
            train_step = optimizer.apply_gradients(grads_and_vars)

            initialize_variables()
            if new_image:
                init_input()
            losses = run_optimization(
                train_step, iterations=iterations, print_n_times=5)
            result_image = input_tensor.eval()
            result_image = np.clip(deprocess(result_image), 0, 255)
            show(result_image)
            return losses, result_image


    # ### Test different optimizers
    print(xx_num)
    string = 'frame' 
    def save_results(optimizer_name, loss=None, result_image=None):
        ''' A helper to save the results of this optimizer run.'''
        global xx_num
        if result_image is not None:
            imsave('results/frame{}.jpg'.format(str(xx_num).zfill(3)), result_image)
            xx_num = xx_num + 1
        if loss is None:
            # Load the loss
            loss = np.loadtxt('results/' + optimizer_name + '.csv')
        else:
            # Save the loss
            loss = np.asarray(loss)
            np.savetxt('results/' + optimizer_name + '.csv', np.asarray(loss))
        plot, = plt.plot(loss[:, 0], loss[:, 1])
        plotHandles.append(plot)
        labels.append(optimizer_name)


    # Turn interactive plotting off
    plt.ioff()
    # To display a legend in the plot
    figure = plt.figure(figsize=(10, 10))
    labels = []
    plotHandles = []
    # We are using a large learning rate to change the initial image sufficiently



    # In[210]:

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # loss, result_image = optimize(optimizer)
    # save_results(optimizer.get_name(), loss, result_image)


    # In[211]:


    # optimizer = tf.train.AdadeltaOptimizer(learning_rate=10)
    # loss, result_image = optimize(optimizer)
    # save_results(optimizer.get_name(), loss, result_image)


    # In[212]:


    # optimizer = tf.train.AdagradOptimizer(learning_rate)
    # loss, result_image = optimize(optimizer)
    # save_results(optimizer.get_name(), loss, result_image)


    # In[213]:


    optimizer = tf.train.AdamOptimizer(learning_rate)
    loss, result_image = optimize(optimizer)
    save_results(optimizer.get_name(), loss, result_image)


    # In[214]:


    # optimizer = tf.train.RMSPropOptimizer(learning_rate)
    # loss, result_image = optimize(optimizer)
    # save_results(optimizer.get_name(), loss, result_image)



    # ### L-BFGS Optimizer
    # In tensorflow L-BFGS is just a wrapper around the python L-BFGS, just itself is 
    # a wrapper around a Fortran implementation of L-BFGS, so the procedure is a bit 
    # different than the other optimizers.

    # In[240]:
    continue
    exit(0)
    # 
    with graph.as_default():
        init_input()
        iterations = ITERAT
        
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            score_op,
            var_list=[input_tensor],
            options={'maxiter': iterations})

        def step_callback(x):
            pass
            # Keep the input_tensor between 0 and 255
            # (gives slightly better output, slows optimization by factor of 2)
    #         x = x.astype('float32').reshape((1,) + image_size + (3,))
    #         x = input_tensor.assign(tf.clip_by_value(x, 0, 255)).eval()
            # Show the image at this iteration
            # show(np.reshape(x, image_size + (3,)))

        def loss_callback(*args):
            wall_time = time.time() - start_time
            losses.append((wall_time, args[0]))

        losses = []
        start_time = time.time()

        optimizer.minimize(
            sess,
            #         feed_dict=[input_tensor],
            fetches=[score_op, style_loss_op, content_loss_op, tv_loss_op],
            step_callback=step_callback,
            loss_callback=loss_callback
        )

        result_image = input_tensor.eval()
        result_image = np.clip(deprocess(result_image), 0, 255)
        show(result_image)
        save_results('L-BFGS', losses, result_image)


    # ### Compare the Optimizers

    # In[216]:


    figure.set_size_inches(20, 10, forward=True)
    plt.legend(plotHandles, labels, prop={'size': 24})
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.tick_params(labelsize=20)
    plt.xlabel('Wall time (seconds)', fontsize=24)
    plt.ylabel('Loss', fontsize=24)
    plt.ylim([0, 1e+7])
    # figure.savefig('graph.jpg')
    # plt.savefig(figure,)



    exit(0)
    with graph.as_default():
        # This might need to be lower.
        learning_rate = 10

        # Recommended at least 500 - 1000 iterations for a good quality image.
        # Good style should be visible even after 100 iters.
        iterations = 500

        # How many times to print the progress
        print_n_times = 5
        print_every_n = max(iterations // print_n_times, 1)

        # To compute total optimization time
        start_time = time.time()

        # To display average losses
        content_losses = []
        style_losses = []
        tv_losses = []
        # Run the optimization
        for i in range(iterations):
            (gradient_computed, score_computed,
            style_loss_computed, content_loss_computed,
            tv_loss_computed) = sess.run(
                [gradient_op, score_op, style_loss_op, content_loss_op,
                tv_loss_op],
                feed_dict={input_tensor: np.extend_dims(result_image,0)})

            # Modify the current image by the gradient
            result_image -= gradient_computed * learning_rate

            # Clip for better quality image
            result_image = np.clip(result_image, 0, 256)

            # Store the losses to display
            content_losses.append(content_loss_computed)
            style_losses.append(style_loss_computed)
            tv_losses.append(tv_loss_computed)

            # Print losses
            if (i % print_every_n == 0):
                print('Iteration: %d Loss: %g  Style L: %g  Content L: %g TV L: %g' %
                    (i,
                    score_computed,
                    np.asarray(style_losses).mean(),
                    np.asarray(content_losses).mean(),
                    np.asarray(tv_losses).mean()))
                content_losses = []
                style_losses = []
                tv_losses = []


    # In[ ]:


    # Show what we got
    show(np.squeeze(deprocess(result_image)))

