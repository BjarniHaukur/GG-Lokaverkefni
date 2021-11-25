import numpy as np
from tensorflow.python.keras.losses import MeanSquaredError
from tqdm import tqdm
import tensorflow as tf
from keras.layers import *
from tensorflow.keras.losses import BinaryCrossentropy
from IPython.display import clear_output

class MyGAN(object):

    # Mostly from https://www.tensorflow.org/tutorials/generative/dcgan

    def __init__(self, ds_train, ds_val, gen_model, disc_model):
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.gen = gen_model
        self.disc = disc_model

        self.gen_train_loss_history = []
        self.disc_train_loss_history = []
        self.gen_train_acc_history = []
        self.disc_train_acc_history = []

        self.gen_val_loss_history = []
        self.disc_val_loss_history = []
        self.gen_val_acc_history = []
        self.disc_val_acc_history = []
        
        self.gen_opt = tf.keras.optimizers.RMSprop()
        self.disc_opt = tf.keras.optimizers.RMSprop()

        self.gan_loss = BinaryCrossentropy(from_logits=True)

        self.pre_loss = MeanSquaredError()

        self.gan_trained = False


    def __discriminator_loss(self, disc_real, disc_fake):
        real_loss = self.gan_loss(tf.ones_like(disc_real), disc_real)
        fake_loss = self.gan_loss(tf.zeros_like(disc_fake), disc_fake)
        return real_loss + fake_loss

    def __generator_loss(self, disc_fake):
        return self.gan_loss(tf.ones_like(disc_fake), disc_fake)


    def __train_step(self, train_batch):
        X_real = train_batch[0]
        y_real = train_batch[1]
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            y_fake = self.gen(X_real)

            disc_real = self.disc(y_real)
            disc_fake = self.disc(y_fake)

            gen_loss  = self.__generator_loss(disc_fake)
            disc_loss = self.__discriminator_loss(disc_real, disc_fake)

        gen_grad  = gen_tape.gradient(gen_loss, self.gen.trainable_variables)
        disc_grad = disc_tape.gradient(disc_loss, self.disc.trainable_variables)

        self.gen_opt.apply_gradients(zip(gen_grad, self.gen.trainable_variables))
        self.disc_opt.apply_gradients(zip(disc_grad, self.disc.trainable_variables))

        return (gen_loss, disc_loss, self.__approx_gen_accuracy(y_real, y_fake), \
                   self.__approx_disc_accuracy(disc_real, disc_fake))


    def __approx_gen_accuracy(self, y_real, y_fake):
        return (y_real==y_fake).mean()

    def __approx_disc_accuracy(self, disc_real, disc_fake):
        return ((disc_real==1).mean() + (disc_fake==0).mean())/2

    def __validation(self, val_batch):
        X_real = val_batch[0]
        y_real = val_batch[1]

        y_fake = self.gen(X_real)

        disc_real = self.disc(y_real)
        disc_fake = self.disc(y_fake)

        gen_loss  = self.__generator_loss(disc_fake)
        disc_loss = self.__discriminator_loss(disc_real, disc_fake)

        gen_acc = self.__approx_gen_accuracy(y_real, y_fake)
        disc_acc = self.__approx_disc_accuracy(disc_real, disc_fake)

        return (gen_loss, disc_loss, gen_acc, disc_acc)
    
    def __log(self, epoch, train_stats, val_stats=None):
        clear_output(wait=True)
        print(f'_____________________________________________ Epoch {epoch+1} _____________________________________________')
        print(f'Generator loss:         {train_stats[0]:.3f}, Generator accuracy:        {train_stats[2]:.3f}')
        if not val_stats is None:
            print(f'Generator val loss:     {val_stats[0]:.3f}, Generator val accuracy:     {val_stats[2]:.3f}')
        print(f'Discriminator loss:     {train_stats[1]:.3f}, Discriminator accuracy:     {train_stats[3]:.3f}')
        if not val_stats is None:
            print(f'Discriminator val loss: {val_stats[1]:.3f}, Discriminator val accuracy: {val_stats[3]:.3f}')
        

    def train_individually(self, num_epochs, num_train_steps):
        if self.gan_trained:
            print("Cannot train generator and discriminator separately after GAN initialization")
        
        self.gen.compile(optimizer='rmsprop', loss='mse')
        self.gen.fit(self.ds_train, epoch=num_epochs, steps_per_epoch=num_train_steps)

        ds_train_iter = self.ds_train.as_numpy_iterator()

        for _ in range(num_epochs):
            ds_next = ds_train_iter.next()
            X_real = ds_next[0]
            y_real = ds_next[1]

            with tf.GradientTape() as disc_tape:
                y_fake = self.gen(X_real)

                disc_real = self.disc(y_real)
                disc_fake = self.disc(y_fake)

                disc_loss = self.discriminator_loss(disc_real, disc_fake)

            disc_grad = disc_tape.gradient(disc_loss, self.disc.trainable_variables)
            self.disc_opt.apply_gradients(zip(disc_grad, self.disc.trainable_variables))

        self.disc_opt = tf.keras.optimizers.RMSprop()


    def fit(self, num_epochs, num_train_steps, num_val_steps):
        self.gan_trained = True
        ds_train_iter = self.ds_train.as_numpy_iterator()
        ds_val_iter   = self.ds_val.as_numpy_iterator()

        for epoch in range(num_epochs):
            train_stats = [0,0,0,0]
            for _ in tqdm(range(num_train_steps)):
                new_stats = self.__train_step(ds_train_iter.next())
                train_stats = [sum(x) for x in zip(train_stats, new_stats)]
            
            train_stats = train_stats/num_train_steps
 
            self.gen_train_loss_history.append(train_stats[0])
            self.disc_train_loss_history.append(train_stats[1])
            self.gen_train_acc_history.append(train_stats[2])
            self.disc_train_acc_history.append(train_stats[3])

            val_stats = [0,0,0,0]
            for _ in range(num_val_steps):
                new_stats = self.__validation(ds_val_iter.next())
                val_stats = [sum(x) for x in zip(val_stats, new_stats)]

            val_stats = val_stats/num_val_steps

            self.gen_train_loss_history.append(val_stats[0])
            self.disc_train_loss_history.append(val_stats[1])
            self.gen_train_acc_history.append(val_stats[2])
            self.disc_train_acc_history.append(val_stats[3])

            self.__log(epoch, train_stats, val_stats)
            

    def get_current_history(self):
        """
        Returns gen_train_loss, disc_train_loss, gen_train_acc, disc_train_acc 
        gen_loss_loss, disc_loss_loss, gen_loss_acc, disc_loss_acc as two tuples
        """
        return (self.gen_train_loss_history, self.disc_train_loss_history, \
               self.gen_train_acc_history, self.disc_train_acc_history), \
               (self.gen_val_loss_history, self.disc_val_loss_history, \
               self.gen_val_acc_history, self.disc_val_acc_history), \






