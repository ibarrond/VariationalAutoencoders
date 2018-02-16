import tensorflow as tf
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tensorflow.examples.tutorials.mnist import input_data

# Progressbar
# bar = progressbar.ProgressBar(widgets=['[', progressbar.Timer(), ']', progressbar.Bar(), '(', progressbar.ETA(), ')'])

# Get the MNIST data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters
results_path = './Adversarial_Autoencoder'

class AAE(object):
    def __init__(self, name,
                 n_inputs=784,
                 n_neurons_encoder=[2048, 256],
                 n_latent=2,
                 n_neurons_decoder=[256, 2048]):
        tf.reset_default_graph()
        
        self.name = name
        self.n_inputs = n_inputs
        self.n_encoder = [n_inputs] + n_neurons_encoder + [n_latent]
        self.n_decoder = [n_latent] + n_neurons_decoder + [n_inputs]
        self.n_latent = n_latent
        self.length_encoder = len(self.n_encoder)
        self.length_decoder = len(self.n_decoder)
        # Placeholders for input data and the targets
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.n_inputs], name='Input')
        self.x_target = tf.placeholder(dtype=tf.float32, shape=[None, self.n_inputs], name='Target')
        self.real_distribution = tf.placeholder(dtype=tf.float32, shape=[None, n_latent], name='Real_distribution')
        self.decoder_input = tf.placeholder(dtype=tf.float32, shape=[1, n_latent], name='Decoder_input')
        
        
        self.encoder_output, self.decoder_output, self.d_real, self.d_fake = self.feedforward()
        self.ae_loss, self.dc_loss, self.gen_loss = self.losses()
        
        
        self.session = tf.Session()


    def form_results(self):
        """
        Forms folders for each run to store the tensorboard files, saved models and the log files.
        :return: three string pointing to tensorboard, saved models and log paths respectively.
        """
        folder_name = "/{0}_Adversarial_Autoencoder". \
            format(datetime.datetime.now())
        tensorboard_path = results_path + folder_name + '/Tensorboard'
        saved_model_path = results_path + folder_name + '/Saved_models/'
        log_path = results_path + folder_name + '/log'
        if not os.path.exists(results_path + folder_name):
            os.mkdir(results_path + folder_name)
            os.mkdir(tensorboard_path)
            os.mkdir(saved_model_path)
            os.mkdir(log_path)
        return tensorboard_path, saved_model_path, log_path


    def generate_image_grid(self, op):
        """
        Generates a grid of images by passing a set of numbers to the decoder and getting its output.
        :param sess: Tensorflow Session required to get the decoder output
        :param op: Operation that needs to be called inorder to get the decoder output
        :return: None, displays a matplotlib window with all the merged images.
        """
        x_points = np.arange(-10, 10, 1.5).astype(np.float32)
        y_points = np.arange(-10, 10, 1.5).astype(np.float32)

        nx, ny = len(x_points), len(y_points)
        plt.subplot()
        gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

        for i, g in enumerate(gs):
            z = np.concatenate(([x_points[int(i / ny)]], [y_points[int(i % nx)]]))
            z = np.reshape(z, (1, 2))
            x = self.session.run(op, feed_dict={self.decoder_input: z})
            ax = plt.subplot(g)
            img = np.array(x.tolist()).reshape(28, 28)
            ax.imshow(img, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('auto')
        plt.show()


    def dense(self, x, n1, n2, name):
        """
        Used to create a dense layer.
        :param x: input tensor to the dense layer
        :param n1: no. of input neurons
        :param n2: no. of output neurons
        :param name: name of the entire dense layer.i.e, variable scope name.
        :return: tensor with shape [batch_size, n2]
        """
        with tf.variable_scope(name, reuse=None):
            xavier_dev = tf.sqrt(2./(n1 + n2))
            weights = tf.get_variable("weights", shape=[n1, n2],
                                      initializer=tf.random_normal_initializer(mean=0., stddev=xavier_dev))
            bias = tf.get_variable("bias", shape=[n2], initializer=tf.random_normal_initializer(mean=0., stddev=xavier_dev))
            out = tf.add(tf.matmul(x, weights), bias, name='matmul')
            return out


    # The autoencoder network
    def encoder(self, x, reuse=False):
        """
        Encode part of the autoencoder.
        :param x: input to the autoencoder
        :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
        :return: tensor which is the hidden latent variable of the autoencoder.
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
        net = x
            
        with tf.name_scope('Encoder'):
            for i in range(self.length_encoder - 1):
                d_in = self.n_encoder[i]
                d_out = self.n_encoder[i+1]
                
                if i < self.length_encoder - 2:
                    # net = tf.layers.batch_normalization(net, training=self.phase)
                    net = tf.nn.tanh(self.dense(net, d_in, d_out, 'e_dense_' + str(i)))
                else:
                    # net = tf.nn.relu(net)
                    net = self.dense(net, d_in, d_out, 'e_latent_variable')
            
            latent_variable = net
            return latent_variable


    def decoder(self, x, reuse=False):
        """
        Decoder part of the autoencoder.
        :param x: input to the decoder
        :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
        :return: tensor which should ideally be the input given to the encoder.
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
            
        net = x
        with tf.name_scope('Decoder'):
            for i in range(self.length_decoder - 1):
                d_in = self.n_decoder[i]
                d_out = self.n_decoder[i+1]

                if i < self.length_encoder - 2:
                    # net = tf.layers.batch_normalization(net, training=self.phase)
                    net = tf.nn.tanh(self.dense(net, d_in, d_out, 'd_dense_' + str(i)))
                else:
                    # net = tf.nn.relu(net)
                    net = tf.nn.sigmoid(self.dense(net, d_in, d_out, 'd_output'))
                    
        output = net
        return output

    def discriminator(self, x, reuse=False):
        """
        Discriminator that is used to match the posterior distribution with a given prior distribution.
        :param x: tensor of shape [batch_size, z_dim]
        :param reuse: True -> Reuse the discriminator variables,
                      False -> Create or search of variables before creating
        :return: tensor of shape [batch_size, 1]
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
            
        net = x
        with tf.name_scope('Discriminator'):
            for i in range(self.length_decoder - 1):
                d_in = self.n_decoder[i]
                d_out = self.n_decoder[i+1]

                if i < self.length_encoder - 2:
                    # net = tf.layers.batch_normalization(net, training=self.phase)
                    net = tf.nn.tanh(self.dense(net, d_in, d_out, 'dc_dense_' + str(i)))
                else:
                    # net = tf.nn.relu(net)
                    net = self.dense(net, d_in, 1, 'dc_output')
                    
        output = net
        return output
        
    def feedforward(self):
        with tf.variable_scope(tf.get_variable_scope()):
            encoder_output = self.encoder(self.x_input)
            decoder_output = self.decoder(encoder_output)
            
        with tf.variable_scope(tf.get_variable_scope()):
            d_real = self.discriminator(self.real_distribution)
            d_fake = self.discriminator(encoder_output, reuse=True)
            
        return encoder_output, decoder_output, d_real, d_fake
    
    def get_ll(self, target, output):
        return tf.reduce_sum(
            target * tf.log(output + 1e-10) + \
            (1 - target) * tf.log(1 - output + 1e-10),
            reduction_indices=[-1]
        )
        
    def losses(self):
        # Autoencoder loss
        # autoencoder_loss = tf.reduce_mean(tf.square(self.x_target - self.decoder_output))
        autoencoder_loss = -tf.reduce_mean(self.get_ll(self.x_target, self.decoder_output))
              
        # Discriminator loss
        dc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_real), logits=self.d_real))
        dc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.d_fake), logits=self.d_fake))
        dc_loss = dc_loss_fake + dc_loss_real        
        
        # Generator loss
        generator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_fake), logits=self.d_fake))        
        
        return autoencoder_loss, dc_loss, generator_loss

    
    def learn(self, learning_rate=0.001, n_epochs=5, batch_size=128):
        """
        Used to train the autoencoder by passing in the necessary inputs.
        :param train_model: True -> Train the model, False -> Load the latest trained model and show the image grid.
        :return: does not return anything
        """

        # Autoencoder loss
        autoencoder_loss = self.ae_loss

        # Discrimminator Loss
        dc_loss = self.dc_loss

        # Generator loss
        generator_loss = self.gen_loss

        all_variables = tf.trainable_variables()
        dc_var = [var for var in all_variables if 'dc_' in var.name]
        en_var = [var for var in all_variables if 'e_' in var.name]

        # Optimizers
        autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(autoencoder_loss)
        discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(dc_loss, var_list=dc_var)
        generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(generator_loss, var_list=en_var)

        init = tf.global_variables_initializer()

        # Reshape immages to display them
        input_images = tf.reshape(self.x_input, [-1, 28, 28, 1])
        generated_images = tf.reshape(self.decoder_output, [-1, 28, 28, 1])

        # Tensorboard visualization
        tf.summary.scalar(name='ae_loss', tensor=autoencoder_loss)
        tf.summary.scalar(name='ds_loss', tensor=dc_loss)
        tf.summary.scalar(name='gen_loss', tensor=generator_loss)
        tf.summary.histogram(name='enc_distribution', values=self.encoder_output)
        tf.summary.histogram(name='real_distribution', values=self.real_distribution)
        tf.summary.image(name='input_images', tensor=input_images, max_outputs=10)
        tf.summary.image(name='gen_images', tensor=generated_images, max_outputs=10)
        summary_op = tf.summary.merge_all()

        # Saving the model
        saver = tf.train.Saver()
        step = 0

        # tensorboard_path, saved_model_path, log_path = self.form_results()
        self.session.run(init)
        # writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=self.session.graph)
        for i in range(n_epochs):
            n_batches = int(mnist.train.num_examples / batch_size)
            print("------------------Epoch {}/{}------------------".format(i, n_epochs))
            for b in range(1, n_batches + 1):
                z_real_dist = np.random.randn(batch_size, self.n_latent) #* 2.5
                batch_x, _ = mnist.train.next_batch(batch_size)
                self.session.run(autoencoder_optimizer, feed_dict={self.x_input: batch_x, self.x_target: batch_x})
                self.session.run(discriminator_optimizer,
                         feed_dict={self.x_input: batch_x, self.x_target: batch_x, self.real_distribution: z_real_dist})
                self.session.run(generator_optimizer, feed_dict={self.x_input: batch_x, self.x_target: batch_x})
                
                """
                if b % 50 == 0:
                    a_loss, d_loss, g_loss, summary = self.session.run(
                        [autoencoder_loss, dc_loss, generator_loss, summary_op],
                        feed_dict={self.x_input: batch_x, self.x_target: batch_x,
                                   self.real_distribution: z_real_dist})
                    writer.add_summary(summary, global_step=step)
                    with open(log_path + '/log.txt', 'a') as log:
                        log.write("Epoch: {}, iteration: {}\n".format(i, b))
                        log.write("Autoencoder Loss: {}\n".format(a_loss))
                        log.write("Discriminator Loss: {}\n".format(d_loss))
                        log.write("Generator Loss: {}\n".format(g_loss))
                step += 1
                """
            a_loss, d_loss, g_loss, summary = self.session.run(
                    [autoencoder_loss, dc_loss, generator_loss, summary_op],
                    feed_dict={self.x_input: batch_x, self.x_target: batch_x,
                               self.real_distribution: z_real_dist})
            print("Autoencoder Loss: {}".format(a_loss))
            print("Discriminator Loss: {}".format(d_loss))
            print("Generator Loss: {}".format(g_loss))
            #saver.save(self.session, save_path=saved_model_path, global_step=step)
            
    def debug(self):
        with tf.variable_scope(tf.get_variable_scope()):
            decoder_image = self.decoder(self.decoder_input, reuse=True)
        # Get the latest results folder
        all_results = os.listdir(results_path)
        all_results.sort()
        saver = tf.train.Saver()
        saver.restore(self.session, save_path=tf.train.latest_checkpoint(results_path + '/' + all_results[-1] + '/Saved_models/'))
        self.generate_image_grid(op=decoder_image)
        
    def benchmark(self, validation=False, batch_size=128):
        if validation:
            benchmark_data = mnist.validation
            label = 'Validation loss:'
        else:
            benchmark_data = mnist.test
            label = 'Test loss:'
        
        cost = 0
        for batch_i in range(benchmark_data.num_examples // batch_size):
            batch_xs, _ = benchmark_data.next_batch(batch_size)
            cost += self.session.run(self.ae_loss,
                                   feed_dict={self.x_input: batch_xs, self.x_target: batch_xs})
        return cost / (benchmark_data.num_examples // batch_size)
    
    def benchmark(self, validation=False, batch_size=128, noisy=False, mean=0, var=0.1):
        """
        Computes validation/test log-likelihood for a trained model.
        It also permits computing the log-likelihood after denoising.
        """
        
        # TEST LOG LIKELIHOOD
        if validation:
            benchmark_data = mnist.validation
        else:
            benchmark_data = mnist.test
        
        total_batch = benchmark_data.num_examples // batch_size
        ell = 0.
        for batch_i in range(total_batch):
            batch_xs, _ = benchmark_data.next_batch(batch_size)
            c = self.session.run(self.ae_loss,
                   feed_dict={self.x_input: batch_xs, self.x_target: batch_xs})
            ell+= c/total_batch
            
        if not noisy:
            return ell
            
        # NOISY TEST LOG LIKELIHOOD
        if validation:
            benchmark_data = mnist.validation
            title = 'Validation LogLikelihood:'
        else:
            benchmark_data = mnist.test
            title = 'Test LogLikelihood:'
        
        total_batch = benchmark_data.num_examples // batch_size
        ell = 0.
        for batch_i in range(total_batch):
            xs, _ = benchmark_data.next_batch(batch_size)
            xs_noisy = np.clip(xs + np.random.normal(mean, var, xs.shape), 0, 1)
            ys_noisy = self.session.run(self.decoder_output,
                   feed_dict={self.x_input: xs_noisy})
            c = self.session.run(self.ae_loss,
                   feed_dict={self.x_target: ys_noisy, self.x_input: xs})
            ell+= c/total_batch
        
        return ell
    
    def predict(self, batch):
        outputs = self.decoder_output
        return self.session.run(outputs, feed_dict={self.x_input: batch, self.x_target: batch})
    
    def serialize(self, path):
        saver = tf.train.Saver()
        save_path = saver.save(self.session, path)
        print("Model saved in file: %s" % save_path)
        
    def restore(self, path):
        saver = tf.train.Saver()   
        sess = tf.Session()
        saver.restore(sess, save_path=path)
        self.session = sess
    
    def plot_enc_dec(self, n_examples=10, save=False):
        """
        Shows n_examples inputs and their reconstructions.
        """
        test_xs = mnist.test.images[0:n_examples]
        recon = self.predict(test_xs)
        fig, axs = plt.subplots(2, n_examples, figsize=(20, 4))
        for example_i in range(n_examples):
            axs[0][example_i].imshow(
                np.reshape(test_xs[example_i, :], (28, 28)))
            axs[1][example_i].imshow(
                np.reshape(
                    np.reshape(recon[example_i, ...], (784,)),
                    (28, 28)))
            plt.gray()
            axs[0][example_i].get_xaxis().set_visible(False)
            axs[0][example_i].get_yaxis().set_visible(False)
            axs[1][example_i].get_xaxis().set_visible(False)
            axs[1][example_i].get_yaxis().set_visible(False)
            
        if(save):
            fig.savefig('images/'+self.name+'_recon.png')
        plt.show()
        
    def plot_latent_repr(self, n_examples=10000, save=False):
        """
        Visualizes the latent space in case 2-dimensional.
        """
        xs = mnist.test.images[0:n_examples]
        zs = self.session.run(self.encoder_output, feed_dict={self.x_input: xs, self.x_target: xs})
        
        fig = plt.figure(figsize=(10, 8))
        
        plt.scatter(zs[:, 0], zs[:, 1], c=np.argmax(ys, 1), marker='o',
                edgecolor='none', cmap=plt.cm.get_cmap('jet', 10))
        plt.colorbar()
        
        if(save):
            fig.savefig('images/'+self.name+'_lat_repr.png')
        
        return fig
    
    def plot_latent_repr(self, n_examples = 10000, save=False, noisy=False, mean=0, var=0.1):
        # Plot manifold of latent layer
        xs = mnist.test.images[0:n_examples]
        # xs = mnist.test.next_batch(n_examples)[0]
        xs_noisy = np.clip(xs + np.random.normal(mean, var, xs.shape), 0 ,1)
        
        if noisy:
            zs = self.session.run(self.encoder_output, feed_dict={self.x_input: xs_noisy, self.x_target: xs})
        else:
            zs = self.session.run(self.encoder_output, feed_dict={self.x_input: xs, self.x_target: xs})
        
        fig = plt.figure(figsize=(10, 8))
        
        plt.scatter(zs[:, 0], zs[:, 1], c=np.argmax(ys, 1), marker='o',
                edgecolor='none', cmap=plt.cm.get_cmap('jet', 10))
        plt.xlim([-6, 6])
        plt.ylim([-6, 6])
        plt.colorbar()
        
        if(save):
            fig.savefig(self.name+'_lat_repr.png')
        
        return fig
    
    def plot_noisy_recon(self, n_examples=20, mean=0, var=0.1, save=False):
        """
        Shows n_examples noisy inputs and their reconstructions.
        """
        
        # xs = mnist.test.next_batch(n_examples)[0]
        xs = mnist.test.images[0:n_examples]
        xs_noisy = np.clip(xs + np.random.normal(mean, var, xs.shape), 0 ,1)
        recon = self.predict(xs_noisy)
        fig, axs = plt.subplots(2, n_examples, figsize=(20, 4))
        for i in range(n_examples):
            axs[0][i].imshow(np.reshape(xs_noisy[i, :], (28, 28)), cmap='gray')
            axs[1][i].imshow(np.reshape(recon[i, ...], (28, 28)), cmap='gray')
            axs[0][i].axis('off')
            axs[1][i].axis('off')
        
        if(save):
            fig.savefig('images/'+self.name+'_noisy_recon.png')
        
        return fig
    
    def plot_latent_recon(self, n_examples=20, l_min=-3, l_max=3, save=False):        
        """
        Visualizes reconstructions from the latent space in an overlay grid.
        """
        
        # Reconstruct Images from equidistant latent representations
        images = []
        fig = plt.figure(figsize=(8,8))
        for img_j in np.linspace(l_max, l_min, n_examples):
            for img_i in np.linspace(l_min, l_max, n_examples):
                z = np.array([[img_i, img_j]], dtype=np.float32)
                recon = self.session.run(self.decoder_output, feed_dict={self.encoder_output: z})
                images.append(np.reshape(recon, (1, 28, 28, 1)))
        images = np.concatenate(images)
        
        # Arrange the images as a square by proximity
        img_n = images.shape[0]
        img_h = images.shape[1]
        img_w = images.shape[2]
        m = np.ones(
            (img_h * n_examples + n_examples + 1,
             img_w * n_examples + n_examples + 1, 3)) * 0.5

        for i in range(n_examples):
            for j in range(n_examples):
                this_filter = i * n_examples + j
                if this_filter < img_n:
                    this_img = images[this_filter, ...]
                    m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                      1 + j + j * img_w:1 + j + (j + 1) * img_w, :] = this_img        
        plt.imshow(m, extent=[l_min,l_max,l_min,l_max])
        
        if(save):
            fig.savefig('images/'+self.name+'_lat_recon.png')
        
        return fig