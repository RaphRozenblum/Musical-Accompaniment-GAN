import glob
import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn


from Generator import Generator
from Discriminator import Discriminator


class HarmonyGAN:
    """
    A GAN Model for music accompaniement, based on MuseGAN. It uses the Generator and Discriminator classes
    defined above
    """

    def __init__(self,
                 grad_weight=10,
                 z_dim=32,
                 batch_size=16,
                 n_bars=3,
                 n_steps_per_bar=96,
                 melody_embed_dim=16,
                 bar_embed_dim=16,
                 ):

        self.name = 'HarmonyGAN'

        self.z_dim = z_dim

        self.n_tracks = 5
        self.n_bars = n_bars
        self.n_steps_per_bar = n_steps_per_bar
        self.n_pitches = 128

        self.input_dim = (self.n_tracks, n_bars, n_steps_per_bar, self.n_pitches)

        self.grad_weight = grad_weight
        self.batch_size = batch_size

        # Keep losses in memory during training:
        self.d_losses = []
        self.g_losses = []

        # Initialize number of epochs trained:
        self.epoch = 0

        # Build Model:
        self.D = Discriminator(self.input_dim).float()
        self.G = Generator(self.batch_size,
                           self.input_dim,
                           self.z_dim,
                           melody_embed_dim,
                           bar_embed_dim).float()

    def _reset_gradient(self):
        self.G_optimizer.zero_grad()
        self.D_optimizer.zero_grad()

    def train(self, dataset, epochs=500, save_every_n_epochs=50, d_loops=1, clamp_weights=0.01, lr_G=0.00005,
              lr_D=0.00005):
        """
        Train the model on the given dataset using Adam optimizer and WGAN algorithm on three G losses
        """

        # Build Optimizers:
        self.G_optimizer = torch.optim.RMSprop(self.G.parameters(), lr=lr_G)
        self.D_optimizer = torch.optim.RMSprop(self.D.parameters(), lr=lr_D)

        # Iterator for epochs. Epochs is added to the number of already-trained epochs.
        tqdm_epochs = tqdm(range(self.epoch, self.epoch + epochs), desc='Training ', unit='epoch', initial=self.epoch,
                           total=self.epoch + epochs)

        for epoch in tqdm_epochs:

            self.epoch += 1

            # Randomly shuffle batches:
            np.random.shuffle(dataset.items)

            tqdm_dataloader = tqdm(range(len(dataset)), desc='D_loss  ... - G_loss  ...', unit='batches', leave=False)
            losses_d_internal = []
            losses_g_internal = []

            for i in tqdm_dataloader:
                try:
                    true_songs = dataset[i]
                except:
                    continue

                # Random permutations within the batch:
                idx = torch.randperm(true_songs.shape[0])
                true_songs = true_songs[idx]
                true_songs = true_songs[:self.batch_size]

                batch_size = true_songs.shape[0]

                # true_song is of shape (batch,n_bars,n_steps_per_bar,n_pitches,n_tracks) and uses int8
                true_songs = true_songs.permute(0, 4, 1, 2, 3).float()
                # Change the range of the notes to [-1,1]:
                true_songs = 2 * (true_songs - 0.5)

                # true_song is of shape (batch,n_tracks,n_bars,n_steps_per_bar,n_pitches)
                true_melodies = true_songs[:, 0, :, :, :].unsqueeze(1)
                # true_melody is of shape (batch,1,n_bars,n_steps_per_bar,n_pitches)

                ####################################################################
                #################### Train Discriminator ###########################
                ####################################################################
                for loop in range(d_loops):
                    # D loops are used to insure that the D losses are not nullified by the slower learning of the generator
                    # This is due to the fact that G has significantly more parameters than D, as is common in GANs

                    # Generate fake song batch:
                    chord_noise = torch.randn(batch_size, 1, self.z_dim, 1)
                    style_noise = torch.randn(batch_size, 1, self.z_dim, 1)
                    tracks_noise = torch.randn(batch_size, self.n_tracks - 1, self.z_dim, 1)
                    groove_noise = torch.randn(batch_size, self.n_tracks - 1, 1, self.z_dim, 1)
                    fake_songs = self.G(chord_noise, style_noise, tracks_noise, groove_noise,
                                        true_melodies)  # Use real melodies to generate the output

                    # Tensors containing labels of either true or fake songs and neutral labels for the random average of images:
                    positive_labels = torch.ones(batch_size, 1)
                    negative_labels = -torch.ones(batch_size, 1)
                    neutral_labels = torch.zeros(batch_size, 1)

                    ################## For True Songs ###############
                    # Forward pass:
                    true_scores = self.D(true_songs)
                    # Compute discriminator Loss:
                    d_loss_real = torch.mean(true_scores)

                    ################# For Fake Songs ################
                    # Forward pass:
                    fake_scores = self.D(fake_songs)
                    # Compute discriminator Loss:
                    d_loss_fake = torch.mean(fake_scores)

                    ################### Total Loss ##################
                    d_loss = -d_loss_real + d_loss_fake
                    # Reset gradients of both G and D optimizers:
                    self._reset_gradient()
                    # Backpropagation on D:
                    d_loss.backward()
                    # Optimize the weights of D:
                    self.D_optimizer.step()

                    # Add loss to list:
                    losses_d_internal.append(d_loss.data.item())

                    # Clamp weights of D:
                    for param in self.D.parameters():
                        param.data.clamp_(-clamp_weights, clamp_weights)

                ####################################################################
                ######################## Train Generator ###########################
                ####################################################################

                # Generate fake song batch:
                chord_noise = torch.randn(batch_size, 1, self.z_dim, 1)
                style_noise = torch.randn(batch_size, 1, self.z_dim, 1)
                tracks_noise = torch.randn(batch_size, self.n_tracks - 1, self.z_dim, 1)
                groove_noise = torch.randn(batch_size, self.n_tracks - 1, 1, self.z_dim, 1)
                fake_songs = self.G(chord_noise, style_noise, tracks_noise, groove_noise,
                                    true_melodies)  # Use real melodies to generate the output

                # Get discriminator's predictions:
                outputs = self.D(fake_songs)

                # Get generator Loss to have D predict the output as a real image:
                g_loss = -torch.mean(outputs)
                losses_g_internal.append(g_loss.data.item())

                # Reset gradient of both G and D:
                self._reset_gradient()

                # Backpropagation on both G and D:
                g_loss.backward()

                # Optimize only the weights of G:
                self.G_optimizer.step()

                # Show losses during training:
                tqdm_dataloader.set_description(
                    desc='D_loss  ' + str(round(np.mean(losses_d_internal), 3)) + ' - G_loss  ' + str(
                        round(np.mean(losses_g_internal), 3)), refresh=True)

            # Average losses across epoch:
            self.d_losses.append(np.mean(losses_d_internal))
            self.g_losses.append(np.mean(losses_g_internal))

            # Save every n epochs:
            if self.epoch % save_every_n_epochs == 0:
                print('Model and losses saved.')
                self.save_model(version='intermediary')

    def _binarize(self, generated, thresh):
        """
        Takes an array of probabilities as an input and returns the binarized version
        """
        return np.where(generated > thresh, 1.0, 0.0)

    def accompaniement(self, conditionnal_track, thresh=0.2):
        """
        Creates an accompaniement for the conditionnal track with random noises
        Returns an array, binarized.

        """
        chord_noise = torch.randn(1, 1, self.z_dim, 1)
        style_noise = torch.randn(1, 1, self.z_dim, 1)
        tracks_noise = torch.randn(1, self.n_tracks - 1, self.z_dim, 1)
        groove_noise = torch.randn(1, self.n_tracks - 1, 1, self.z_dim, 1)
        generated = self.G(chord_noise, style_noise, tracks_noise, groove_noise, conditionnal_track).data.numpy()
        return self._binarize(generated, thresh)

    def show_losses(self, directory='', show=True):
        """
        Displays all losses and saves them in the running directory
        """

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # plt.plot([x[0] for x in self.d_losses], label='Critic loss on real scores', alpha=0.7)
        plt.plot(self.d_losses, label='Critic loss', alpha=1)
        # plt.plot([x[1] for x in self.d_losses], label='Critic loss on generated scores', alpha=0.7)
        # plt.plot([x[2] for x in self.d_losses], label='Critic Partial loss (avg scores)', alpha=0.7)
        plt.plot(self.g_losses, label='Generator loss', alpha=1)

        ax.legend(loc='upper right')

        plt.xlabel('Epochs', fontsize=18)
        plt.ylabel('loss', fontsize=16)

        plt.savefig(directory + 'losses.png')

        if show:
            plt.show()

    def save_model(self, version='final'):
        """
        Method to save the model either at the end of the run or at a certain epoch
        """

        if not os.path.isdir('Models'):
            os.mkdir('Models')
            os.mkdir('Models/final/')
            os.mkdir('Models/intermediary/')

        if version == 'final':
            directory = 'Models/final/'
            subdirs = glob.glob(directory + '*/')
            last_run_number = -1
            for d in subdirs:
                run_number = int(d.split('_')[-1][:-1])
                if run_number > last_run_number:
                    last_run_number = run_number
            directory = directory + 'Model_' + str(last_run_number + 1)
            os.mkdir(directory)

        elif version == 'intermediary':
            directory = 'Models/intermediary/'
            subdirs = glob.glob(directory + '*/')

            directory = directory + 'Epoch_' + str(self.epoch)
            os.mkdir(directory)

        # Save models:
        torch.save(self.G, directory + '/Generator.ckpt')
        torch.save(self.D, directory + '/Discriminator.ckpt')
        torch.save(self.G.bar_encoder, directory + '/BarEncoder.ckpt')
        torch.save(self.G.melody_encoder, directory + '/MelodyEncoder.ckpt')
        torch.save(self.G.chord_temporal_network, directory + '/ChordsTemporalNetwork.ckpt')
        for i in range(self.n_tracks - 1):
            torch.save(self.G.track_temporal_networks[i], directory + '/TracksTemporalNetwork_' + str(i + 1) + '.ckpt')
            torch.save(self.G.track_bar_generators[i], directory + '/BarGenerator_' + str(i + 1) + '.ckpt')

        # Save losses:
        self.show_losses(directory=directory + '/', show=False)
