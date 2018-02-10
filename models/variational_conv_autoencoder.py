import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import time

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def lrelu(x, leak=0.2, name="lrelu"):
    """Leaky rectifier.
    Parameters
    ----------
    x : Tensor
        The tensor to apply the nonlinearity to.
    leak : float, optional
        Leakage parameter.
    name : str, optional
        Variable scope to use.
    Returns
    -------
    x : Tensor
        Output of the nonlinearity.
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

class VariationalConvAutoencoder(object):
    def __init__(self, name,
                 n_inputs=784,
                 n_conv_maps = [10, 10, 10],
                 n_conv_filters = [3, 3, 3], 
                 n_conv_strides = [2, 2, 2], 
                 n_conv_padding = ['SAME', 'VALID', 'SAME'], 
                 n_dense = [256, 128],
                 n_latent=2,
                 activation = lrelu):
        
        tf.reset_default_graph()
        
        self.name = name
        print("VCAE "+self.name)
        
        # ATTRIBUTES
        self.N = n_inputs
        self.n_conv_m = n_conv_maps
        self.n_conv_f = n_conv_filters
        self.n_conv_s = n_conv_strides 
        self.n_conv_p = n_conv_padding
        self.n_dense = n_dense
        self.n_latent = n_latent
        self.act = activation
        
        ## DATA PLACEHOLDERS (BATCHES)
        with tf.name_scope('input'):
            self.X = tf.placeholder(tf.float32, shape=[None, self.N], name='X')
            
        with tf.name_scope('input_reshape'):
            self.image_X = tf.reshape(self.X, [-1, 28, 28, 1])
               
        self.X_size = tf.shape(self.X)[0]
            
        ## COMPUTATIONAL GRAPH
        self.Y, self.z_mu, self.z_log_sigma, self.z = self.feedforward()
        self.loss, self.kl, self.ell = self.get_nelbo()

        ## Initialize the session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        
    ## ---------------------------------------------------------------------            
    ## --------------- TF WEIGHTS & BIASES INITIALIZATION ------------------
    ## ---------------------------------------------------------------------
    def xav(self, shape):
        """
        Xavier initialization for the weights and biases
        """
        initial = tf.random_normal(shape, mean=0.0,
                               stddev=tf.sqrt(3./sum(shape)))
        return tf.Variable(initial)    

    
    ## ---------------------------------------------------------------------            
    ## ----------------- SAMPLING FROM THE LATENT SPACE --------------------
    ## ---------------------------------------------------------------------
    
    def get_samples(self, d_in, d_out):
        """
        Sample from noise distribution p(eps) ~ N(0, 1)
        a matrix of samples of dimension [d_in, d_out].
        """
        return tf.random_normal(shape=[d_in, d_out])

    def sample_from_Z(self, z_mu, z_log_o):
        """
        Samples from the posterior of the variational latent space.
        We draw samples using the reparameterization trick.
        """
        return z_mu + tf.exp(z_log_o) * self.get_samples(self.X_size, self.n_latent)
    
        
        
    ## ---------------------------------------------------------------------            
    ## --------------------------- FEEDFORWARD -----------------------------
    ## ---------------------------------------------------------------------
    def encoder(self, net):
        c_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        d_init = tf.contrib.layers.xavier_initializer(uniform=False)
        '''ENCODER: transform the input image into the latent space'''
        # Convolutional Layers
        print(' --> Encoder')
        print(net.shape)
        for i in range(len(self.n_conv_m)):
            net = tf.layers.conv2d(net,
                filters = self.n_conv_m[i],
                kernel_size = [self.n_conv_f[i], self.n_conv_f[i]],
                strides =     [self.n_conv_s[i], self.n_conv_s[i]],
                activation = self.act,
                padding = self.n_conv_p[i], 
                kernel_initializer = c_init)
            print(net.shape)
        self.conv_shape = tf.shape(net)
        
        # Dense layers    
        net = tf.contrib.layers.flatten(net)
        self.flat_shape = net.shape
        for i in range(len(self.n_dense)):
            print(net.shape)
            net = tf.layers.dense(net, self.n_dense[i], activation=self.act,
                                  kernel_initializer=d_init)
            
        print(net.shape)
        # Latent Weights   
        print(' --> Latent space') 
        z_mu = tf.layers.dense(net, self.n_latent,
                               kernel_initializer=d_init)
        z_log_sigma = 0.5 * tf.layers.dense(net, self.n_latent, 
                               kernel_initializer=d_init)
        
        return z_mu, z_log_sigma
        
        
    def decoder(self, net):
        '''DECODER: transform a Latent Space representation into an image'''
        c_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
        d_init = tf.contrib.layers.xavier_initializer(uniform=False)
        # Dense layers
        print(net.shape)
        print(' --> Decoder') 
        for i in reversed(range(len(self.n_dense))):
            net = tf.layers.dense(net, self.n_dense[i], activation=self.act,
                                  kernel_initializer=d_init)
            print(net.shape)
        net = tf.layers.dense(net, self.flat_shape[1], activation=self.act, 
                             kernel_initializer=d_init)    
        print(net.shape)
        net = tf.reshape(net, self.conv_shape)
        print(net.shape)
        # Convolutional Layers
        for i in reversed(range(1, len(self.n_conv_m))):
            net = tf.layers.conv2d_transpose(net,
                filters = self.n_conv_m[i],
                kernel_size = [self.n_conv_f[i], self.n_conv_f[i]],
                strides =     [self.n_conv_s[i], self.n_conv_s[i]],
                activation = self.act,
                padding=self.n_conv_p[i],
                kernel_initializer=c_init)
            print(net.shape)
        net = tf.layers.conv2d_transpose(net,
            filters = 1,
            kernel_size = [self.n_conv_f[0], self.n_conv_f[0]],
            strides =     [self.n_conv_s[0], self.n_conv_s[0]],
            activation = self.act,
            padding=self.n_conv_p[0],
            kernel_initializer=c_init)
        print(net.shape)
        net = tf.contrib.layers.flatten(net)        
        print(net.shape)
        return tf.sigmoid(net)
                
    
    def feedforward(self):
        """
        Feedforward pass excluding last layer's transfer function.
        intermediate : index of intermediate layer for output generation
        """
        net = self.image_X

        # ENCODER
        z_mu, z_log_sigma = self.encoder(net)

        # LATENT: Sample from posterior
        z = self.sample_from_Z(z_mu, z_log_sigma)

        # DECODER
        Y = self.decoder(z)        
                
        return Y, z_mu, z_log_sigma, z
    
    
    ## ---------------------------------------------------------------------            
    ## ------------------- LOSS: EXPECTED LOWER BOUND ----------------------
    ## ---------------------------------------------------------------------    
    
    def get_ell(self):
        """
        Returns the expected log-likelihood of the lower bound.
        For this we use a bernouilli LL.
        """
        target = self.X
        output = self.Y
        # p(x|z)        
        return - tf.reduce_sum((  target  ) * tf.log(output + 1e-10) +
                               (1 - target) * tf.log(1 - output + 1e-10), -1)

    
    def get_kl(self):
        """
        d_kl(q(z|x)||p(z)) returns the KL-divergence between the prior p and the variational posterior q.
        :return: KL divergence between q and p
        """   
        # Formula: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return - 0.5 * tf.reduce_sum( 1.0 + 2.0 * self.z_log_sigma - tf.square(self.z_mu) -
                                   tf.exp(2.0 * self.z_log_sigma), 1)
        
        
    def get_nelbo(self):
        """ Returns the negative ELBOW, which allows us to minimize instead of maximize. """
        kl = tf.reduce_mean(self.get_kl())
        ell = tf.reduce_mean(self.get_ell())
        nelbo = kl + ell
        return nelbo, kl, ell
    
    
    ## ---------------------------------------------------------------------            
    ## --------------------------- LEARNING --------------------------------
    ## ---------------------------------------------------------------------  
    
    def learn(self, learning_rate=0.001, batch_size=128,
                    epochs=100, display_step=1):
        """ Our learning procedure """
        optimizer = tf.train.AdamOptimizer(learning_rate)

        ## Set all_variables to contain the complete set of TF variables to optimize
        all_variables = tf.trainable_variables()

        ## Define the optimizer
        train_step = optimizer.minimize(self.loss, var_list=all_variables)

        tf.summary.scalar('negative_elbo', self.loss)
        #tf.summary.scalar('kl_div', self.kl)
        #tf.summary.scalar('ell', self.ell)
        
        merged = tf.summary.merge_all()
        
        train_writer = tf.summary.FileWriter('logs/train', self.session.graph)
        test_writer = tf.summary.FileWriter('logs/test')        
        
        ## Initialize all variables
        self.session.run(tf.global_variables_initializer())
        
        # Initial model print
        print("*MODEL [", self.name,"] {l_r: %.4f; n_iter: %d; batch: %d}"%\
              (learning_rate, epochs, batch_size))
        print (" -> START Training!")
        
        t = time.time()
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for epoch in range(epochs):
            train_cost = 0.
            batch_ell = 0.
            batch_kl = 0.
            for batch_i in range(total_batch):
                batch_xs, _ = mnist.train.next_batch(batch_size)

                _, loss, ell, kl, summary = self.session.run(
                    [train_step, self.loss, self.ell, self.kl, merged],
                                    feed_dict={self.X: batch_xs})
                
                train_writer.add_summary(summary, epoch * total_batch + batch_i)
                train_cost += loss / total_batch
                batch_ell += ell / total_batch
                batch_kl += kl / total_batch
                
            print("   [%.1f] Epoch: %02d | NELBO: %.6f | ELL: %.6f | KL: %.6f"%(
                time.time()-t,epoch, train_cost, batch_ell, batch_kl ))
        
        print (" -> Training FINISHED in %.1f seconds."%(time.time()-t))
        self.serialize('DUMPS/')
        train_writer.close()
        test_writer.close()
        
    def benchmark(self, validation=False, batch_size = 128):
        # TEST LOG LIKELIHOOD
        if validation:
            benchmark_data = mnist.validation
            title = 'Validation LogLikelihood:'
        else:
            benchmark_data = mnist.test
            title = 'Test LogLikelihood:'
        
        total_batch = int(mnist.train.num_examples/batch_size)
        ell = 0.
        for batch_i in range(total_batch):
            batch_xs, _ = benchmark_data.next_batch(batch_size)
            batch_ell = self.session.run(self.ell,
                   feed_dict={self.X: batch_xs})
            ell+= batch_ell/total_batch
        print("Model ", self.name, ". ", title, ell)
        
        # RECONSTRUCTION EXAMPLES
        f1 = self.plot_recon(save=True)
        
        # NOISY RECONSTRUCTION EXAMPLES
        f2 = self.plot_noisy_recon(save=True)
        
        # LATENT SPACE RECONSTRUCTION
        f3=None
        if(self.n_latent==2):
            f3 = self.plot_latent_recon(save=True)
        
        # LATENT SPACE SCATTER
        f4=None
        if(self.n_latent==2):
            f4 = self.plot_latent_repr(save=True)
        
        return ell, f1, f2, f3, f4
        
    def serialize(self, path):
        '''Save the model in a file'''
        
        saver = tf.train.Saver()
        save_path = saver.save(self.session, path+self.name)
        print("Model saved in file: %s" % path+self.name)
        
        
    def restore(self, path):
        '''Restore the saved model'''
        
        saver = tf.train.Saver()   
        sess = tf.InteractiveSession()
        saver.restore(sess, save_path=path)
        self.session = sess
        print("Model restored from file: %s" % path)
    
    def encode(self, input_vector):
        '''Encode the input into the Latent Space'''
        
        z_m, z_l_o, z = self.session.run([self.z_mu, self.z_log_sigma, self.z],
                                feed_dict={self.X: input_vector})
        return z, z_m, z_l_o
    
    def decode(self, z):
        '''Decode from the latent space into the out'''
        y = self.session.run(self.Y,
                                feed_dict={self.z: z})
        return z
  
    def plot_recon(self, n_examples=20, save=False):
        '''Visualize Example Reconstrutions for the model'''
        
        xs = mnist.test.next_batch(n_examples)[0]
        recon = self.session.run(self.Y, feed_dict={self.X: xs})
        fig, axs = plt.subplots(2, n_examples, figsize=(20, 4))
        for i in range(n_examples):
            axs[0][i].imshow(np.reshape(xs[i, :], (28, 28)), cmap='gray')
            axs[1][i].imshow(np.reshape(recon[i, ...], (28, 28)), cmap='gray')
            axs[0][i].axis('off')
            axs[1][i].axis('off')
        
        if(save):
            fig.savefig(self.name+'_recon.png')
        
        return fig
    
    
    def plot_noisy_recon(self, n_examples=20, mean=0, var=0.1, save=False):
        '''Visualize Example Reconstrutions for the model'''
        
        xs = mnist.test.next_batch(n_examples)[0]
        xs_noisy = xs + np.random.normal(mean, var, xs.shape)
        recon = self.session.run(self.Y, feed_dict={self.X: xs_noisy})
        fig, axs = plt.subplots(2, n_examples, figsize=(20, 4))
        for i in range(n_examples):
            axs[0][i].imshow(np.reshape(xs_noisy[i, :], (28, 28)), cmap='gray')
            axs[1][i].imshow(np.reshape(recon[i, ...], (28, 28)), cmap='gray')
            axs[0][i].axis('off')
            axs[1][i].axis('off')
        
        if(save):
            fig.savefig(self.name+'_noisy_recon.png')
        
        return fig
    
    
    
    def plot_latent_recon(self, n_examples=20, l_min=-3, l_max=3, save=False):        
        '''Visualize Reconstructions from the latent space'''
        
        # Reconstruct Images from equidistant latent representations
        images = []
        fig = plt.figure(figsize=(8,8))
        for img_j in np.linspace(l_max, l_min, n_examples):
            for img_i in np.linspace(l_min, l_max, n_examples):
                z = np.array([[img_i, img_j]], dtype=np.float32)
                recon = self.session.run(self.Y, feed_dict={self.z: z,
                                self.X: [mnist.test.images[0]]})
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
            fig.savefig(self.name+'_lat_recon.png')
        
        return fig
    
    def plot_latent_repr(self, n_examples = 10000, save=False):
        # Plot manifold of latent layer
        xs, ys = mnist.test.next_batch(n_examples)
        zs = self.session.run(self.z, feed_dict={self.X: xs})
        
        fig = plt.figure(figsize=(10, 8))
        
        plt.scatter(zs[:, 0], zs[:, 1], c=np.argmax(ys, 1), marker='o',
                edgecolor='none', cmap=plt.cm.get_cmap('jet', 10))
        plt.xlim([-6, 6])
        plt.ylim([-6, 6])
        plt.colorbar()
        
        if(save):
            fig.savefig(self.name+'_lat_repr.png')
        
        return fig
    
