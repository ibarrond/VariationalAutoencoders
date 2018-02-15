import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class BayesianVAEArtificial(object):
    """
    Bayesian Auto-Encoder with gaussian intermediate sampling.
    """
    def __init__(self, name,
                 n_inputs=784,
                 n_neurons_encoder=[2048, 256],
                 n_latent=2,
                 n_neurons_decoder=[256, 2048],
                 constant_prior=False):
        # SIZES
        tf.reset_default_graph()
        
        self.name = name
        self.n_inputs = n_inputs
        self.n_encoder = n_neurons_encoder + [n_latent]
        self.n_decoder = n_neurons_decoder + [n_inputs]
        self.length_encoder = len(self.n_encoder)
        self.length_decoder = len(self.n_decoder)
        self.layers = self.length_encoder + self.length_decoder
        ## Set the number of Monte Carlo samples as a placeholder so that it can be different for training and test
        self.N = tf.placeholder(tf.int32)
        self.L = tf.placeholder(tf.int32)
        
        self.phase = tf.placeholder(tf.bool, name='phase')
        self.constant_prior = constant_prior
        
        ## Batch data placeholders
        with tf.name_scope('input'):
            self.X = tf.placeholder(tf.float32, shape=[None, self.n_inputs], name='X')
        
        self.prior_mean_W, self.log_prior_var_W, self.mean_W, self.log_var_W = self.initialize_W()
        
        ## Initialize the session
        self.session = tf.Session()
        
        ## Builds whole computational graph with relevant quantities as part of the class
        self.Y, self.Y_exp, self.z, self.z_exp, self.z_mu, self.z_log_sigma = self.feedforward()
        self.loss, self.kl, self.ell = self.get_nelbo()

        
    def create_weight_variable(self, shape, is_prior=False):
        """
        We initialize the weight distributions using xavier initialization.
        Even the means are sampled from a truncated normal distribution with xavier variance.
        """
        if self.constant_prior and is_prior:
            mean = tf.constant(0.0, shape=shape)
            log_var = tf.constant(0.0, shape=shape)
        else:
            # xavier initialization
            log_var = tf.Variable(tf.ones(shape)) * tf.log(2./(shape[-1] + shape[-2]))
            mean = tf.Variable(tf.truncated_normal(shape, stddev=tf.sqrt(2./(shape[-1] + shape[-2]))))
            
        tf.summary.histogram('mean', mean)
        tf.summary.histogram('logvar', log_var)
            
        return mean, log_var
    
    def initialize_W(self):
        """
        Define a prior and a posterior weight distribution.
        We assume both to be standard normal iid.
        """
        prior_means = []
        prior_log_vars = []
        post_means = []
        post_log_vars = []
        
        with tf.name_scope("encoder_layer_weights"):
            for i in range(self.length_encoder):
                if i == 0:
                    d_in, d_out = (self.n_inputs, self.n_encoder[0])
                else:
                    d_in, d_out = (self.n_encoder[i-1], self.n_encoder[i])
                    
                d_in += 1 # account for the bias weight
                    
                with tf.name_scope("priors"):
                    prior_mean, prior_log_var = self.create_weight_variable([d_in, d_out], is_prior=True)
                    # prior_mean, prior_log_var = self.create_weight_variable([1], is_prior=True)
                    prior_means.append(prior_mean)
                    prior_log_vars.append(prior_log_var)
                with tf.name_scope("posts"):
                    post_mean, post_log_var = self.create_weight_variable([d_in, d_out])
                    post_means.append(post_mean)
                    post_log_vars.append(post_log_var)
                
        with tf.name_scope("decoder_layer_weights"):
            for i in range(self.length_decoder):
                if i == 0:
                    d_in, d_out = (self.n_encoder[-1], self.n_decoder[0])
                else:
                    d_in, d_out = (self.n_decoder[i-1], self.n_decoder[i])
                    
                d_in += 1 # account for the bias weight
                
                with tf.name_scope("priors"):
                    prior_mean, prior_log_var = self.create_weight_variable([d_in, d_out], is_prior=True)
                    # prior_mean, prior_log_var = self.create_weight_variable([1], is_prior=True)
                    prior_means.append(prior_mean)
                    prior_log_vars.append(prior_log_var)
                with tf.name_scope("posts"):
                    post_mean, post_log_var = self.create_weight_variable([d_in, d_out])
                    post_means.append(post_mean)
                    post_log_vars.append(post_log_var)
        
        return prior_means, prior_log_vars, post_means, post_log_vars
    
    def get_std_norm_samples(self, shape):
        """
        Draws N(0,1) samples of dimension [d_in, d_out].
        """
        return tf.random_normal(shape=shape)

    def sample_from_W(self, mean_W, log_var_W):
        """
        Samples from the variational posterior approximation.
        We draw W-samples for each layer using the reparameterization trick.
        """

        mc_samples = self.L
        d_in, d_out = mean_W.shape
        z = self.get_std_norm_samples([mc_samples, d_in, d_out])
        ## division by 2 to obtain pure standard deviation
        w_from_q = tf.add(tf.multiply(z, tf.exp(log_var_W / 2)), mean_W)

        return w_from_q
    
    def encode(self, net):
        """
        Maps the inputs into the latent representation.
        Additionaly we compute the sample mean and sample variance over the monte-carlo samples.
        """
        batch_size = tf.shape(self.X)[0]
        
        with tf.name_scope("encoder"):
            for i in range(self.length_encoder):
                W = self.sample_from_W(self.mean_W[i], self.log_var_W[i])
                net = tf.matmul(net, W[:,1:,:])
                bias = tf.expand_dims(W[:,0,:], 1)
                net = net + bias
                tf.summary.histogram('activations_l_' + str(i), net)
                # don't tanh z, otherwise limited to -1,1
                if i < self.length_encoder - 1:
                    # net = tf.layers.batch_normalization(net, training=self.phase)
                    net = tf.nn.tanh(net)
                else:
                    # net = tf.nn.relu(net)
                    net = net
                    
                    mean, var = tf.nn.moments(net, axes=[0])
                    net = tf.random_normal([self.L, batch_size, tf.shape(net)[-1]])
                    net = tf.add(tf.multiply(tf.sqrt(var), net), mean)
                    
                tf.summary.histogram('outputs_l_' + str(i), net)
        
        return net, mean, tf.log(var)
    
    def decode(self, net):
        """
        Decodes the latent representation and reconstructs the output.
        """
        w_off = self.length_encoder
        with tf.name_scope("decoder"):
            for i in range(w_off, w_off + self.length_decoder):
                W = self.sample_from_W(self.mean_W[i], self.log_var_W[i])
                net = tf.matmul(net, W[:,1:,:])
                bias = tf.expand_dims(W[:,0,:], 1)
                net = net + bias
                
                # net = tf.layers.batch_normalization(net, training=self.phase)
                
                tf.summary.histogram('activations_l_' + str(i-w_off), net)

                if i == (w_off + self.length_decoder - 1):
                    net = tf.nn.sigmoid(net)
                else:
                    net = tf.nn.tanh(net)
                tf.summary.histogram('outputs_l_' + str(i-w_off), net)

        return net
    
    def feedforward(self):
        """
        Defines the whole feedforward pass through the autoencoder.
        """
        # We will generate L output samples
        batch_size = tf.shape(self.X)[0]
        d_in = tf.shape(self.X)[-1]
        net = tf.multiply(tf.ones([self.L, batch_size, d_in]), self.X)
        
        z, z_mu, z_log_sigma = self.encode(net)
        z_exp = tf.reduce_mean(z, 0)
        
        Y = self.decode(z)
        Y_exp = tf.reduce_mean(Y, 0)
        
        return Y, Y_exp, z, z_exp, z_mu, z_log_sigma
    
    
    """
    def get_ll(self, Y, output):
        batch_size = tf.shape(Y)[0]
        d_in = tf.shape(Y)[-1]
        Y = tf.multiply(tf.ones([self.L, batch_size, d_in]), Y)
        return -tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output)
    """
        
    def get_ll(self, target, output):
        """
        Bernoulli Log-Likelihood.
        """
        return tf.reduce_sum(
            target * tf.log(output + 1e-10) + \
            (1 - target) * tf.log(1 - output + 1e-10),
            reduction_indices=[-1]
        )
    
    def get_ell(self, intermediate=0):
        """
        Returns the expected log-likelihood of the lower bound.
        For this we draw L samples from W, compute the log-likelihood for each
        and average the log-likelihoods in the end (expectation approximation).
        """
        
        # outputs, _, _, _ = self.feedforward()
        outputs = self.Y
        ll = self.get_ll(self.X, outputs)
        ell = tf.reduce_mean(ll, 0)

        return ell

    def get_kl(self, mean_W, log_var_W, prior_mean_W, log_prior_var_W):
        """
        KL[q || p] returns the KL-divergence between the prior p and the variational posterior q.
        :param mq: vector of means for q
        :param log_vq: vector of log-variances for q
        :param mp: vector of means for p
        :param log_vp: vector of log-variances for p
        :return: KL divergence between q and p
        """
        mq = mean_W
        log_vq = log_var_W
        mp = prior_mean_W
        log_vp = log_prior_var_W
        
        return 0.5 * tf.reduce_sum(log_vp - log_vq + (tf.square(mq - mp) / tf.exp(log_vp)) + tf.exp(log_vq - log_vp) - 1)
    
    def get_kl_z(self):
        """
        d_kl(q(z|x)||p(z)) returns the KL-divergence between the gaussian parameters of z and N(0,1).
        We take the average divergence over the batch samples.
        """   
        # Formula: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return -tf.reduce_mean(0.5 * tf.reduce_sum( 1.0 + 2.0 * self.z_log_sigma - tf.square(self.z_mu) -
                                   tf.exp(2.0 * self.z_log_sigma), 1))

    def get_kl_multi(self):
        """
        Compute KL divergence between variational posterior and prior using a multi-layer-network.
        This includes the weights as well as the paramters of the latent distribution.
        """
        kl = 0
        
        for i in range(self.layers - 1):
            kl = kl + self.get_kl(
                        self.mean_W[i],
                        self.log_var_W[i],
                        self.prior_mean_W[i],
                        self.log_prior_var_W[i]
            )
            
        kl += self.get_kl_z()
        
        return kl
    
    def get_nelbo(self):
        """
        Returns the negative ELBOW, which allows us to minimize instead of maximize.
        """
        batch_size = tf.cast(tf.shape(self.X)[0], "float32")
        # the kl does not change among samples
        kl = self.get_kl_multi()
        ell = self.get_ell()
        # batch_ell = tf.reduce_mean(tf.reduce_sum(ell, [-1]))
        batch_ell = tf.reduce_mean(ell)
        nelbo = kl - tf.reduce_sum(ell) * tf.cast(self.N, "float32") / batch_size
        # nelbo = kl - batch_ell
        # nelbo = -batch_ell
        return nelbo, kl, batch_ell
    
    def learn(self, learning_rate=0.01, epochs=50, batch_size=128, mc_samples=10):
        """
        Our learning procedure.
        """
        optimizer = tf.train.AdamOptimizer(learning_rate)

        ## Set all_variables to contain the complete set of TF variables to optimize
        all_variables = tf.trainable_variables()

        ## Define the optimizer
        train_step = optimizer.minimize(self.loss, var_list=all_variables)

        tf.summary.scalar('negative_elbo', self.loss)
        tf.summary.scalar('kl_div', self.kl)
        tf.summary.scalar('ell', self.ell)
        
        merged = tf.summary.merge_all()
        
        train_writer = tf.summary.FileWriter('logs/train', self.session.graph)
        test_writer = tf.summary.FileWriter('logs/test')        
        
        ## Initialize all variables
        init = tf.global_variables_initializer()

        ## Initialize TF session
        self.session.run(init)
        
        num_batches = mnist.train.num_examples // batch_size

        start_time = time.time()
        
        for i in range(epochs):
            train_cost = 0
            cum_ell = 0
            cum_kl = 0
            
            old_progress = 0
            for batch_i in range(num_batches):
                progress = round(float(batch_i) / num_batches * 100)
                if progress % 10 == 0 and progress != old_progress:
                    # print('Progress: ', str(progress) + '%')
                    old_progress = progress

                batch_xs, _ = mnist.train.next_batch(batch_size)

                _, loss, ell, kl, summary = self.session.run(
                    [train_step, self.loss, self.ell, self.kl, merged],
                    feed_dict={self.X: batch_xs, self.L: mc_samples, self.N: mnist.train.num_examples})
                train_writer.add_summary(summary, i)
                train_cost += loss/num_batches
                cum_ell += ell/num_batches
                cum_kl += kl/num_batches

            val_cost = self.benchmark(validation=True)
            
            print("   [%.1f] Epoch: %02d | NELBO: %.6f | ELL: %.6f | KL: %.6f | Val. ELL: %.6f"%(
                time.time()-start_time,i+1, train_cost, cum_ell, cum_kl, val_cost ))        
        
        train_writer.close()
        test_writer.close()
        
        print("Total training time: ", time.time() - start_time)
    
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
            c = self.session.run(self.ell,
                   feed_dict={self.X: batch_xs, self.L: 10, self.N: benchmark_data.num_examples})
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
            xs_noisy = np.clip(xs + np.random.normal(mean, var, xs.shape), 0 ,1)
            ys_noisy = self.session.run(self.Y,
                   feed_dict={self.X: xs_noisy, self.L: 10, self.N: benchmark_data.num_examples})
            c = self.session.run(self.ell,
                   feed_dict={self.Y: ys_noisy, self.X: xs, self.L: 10, self.N: benchmark_data.num_examples})
            ell+= c/total_batch
        
        return ell
        
    def serialize(self, path):
        """
        Saves the model.
        """
        saver = tf.train.Saver()
        save_path = saver.save(self.session, path)
        print("Model saved in file: %s" % save_path)
        
    def restore(self, path):
        """
        Restores a saved model.
        """
        saver = tf.train.Saver()   
        sess = tf.Session()
        saver.restore(sess, save_path=path)
        self.session = sess
    
    def predict(self, batch):
        """
        Reconstructs a batch of inputs.
        """
        outputs = self.Y
        return self.session.run(self.Y_exp, feed_dict={self.X: batch, self.L: 10, self.N: 1, self.phase: False})
    
    def get_weights(self):
        """
        Returns all weights for debugging.
        """
        weights = (self.prior_mean_W, self.log_prior_var_W, self.mean_W, self.log_var_W)
        return self.session.run(weights)
    
    def plot_enc_dec(self, n_examples=10, save=False):
        """
        Shows n_examples inputs and their reconstructions.
        """
        # Plot example reconstructions
        test_xs, _ = mnist.test.next_batch(n_examples)
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
        
    def plot_noisy_recon(self, n_examples=20, mean=0, var=0.1, save=False):
        """
        Shows n_examples noisy inputs and their reconstructions.
        """
        
        xs = mnist.test.next_batch(n_examples)[0]
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
                z = np.array([[[img_i, img_j]]], dtype=np.float32)
                recon = self.session.run(self.Y_exp, feed_dict={self.z: z, self.L: 1})
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
    
    def plot_latent_repr(self, n_examples = 10000, save=False):
        """
        Visualizes the latent space in case 2-dimensional.
        """
        # Plot manifold of latent layer
        xs, ys = mnist.test.next_batch(n_examples)
        zs = self.session.run(self.z_exp, feed_dict={self.X: xs, self.L: 10})
        
        fig = plt.figure(figsize=(10, 8))
        
        plt.scatter(zs[:, 0], zs[:, 1], c=np.argmax(ys, 1), marker='o',
                edgecolor='none', cmap=plt.cm.get_cmap('jet', 10))
        plt.colorbar()
        
        if(save):
            fig.savefig('images/'+self.name+'_lat_repr.png')
        
        return fig
