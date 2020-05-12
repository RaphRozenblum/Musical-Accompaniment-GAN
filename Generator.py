import torch
import torch.nn as nn

from TemporalNetwork import TemporalNetwork
from BarEncoder import BarEncoder
from MelodyEncoder import MelodyEncoder
from BarGenerator import BarGenerator

class Generator(nn.Module):
    """
    The Generator of the model
    """

    def __init__(self, batch_size, output_shape, z_dim, melody_output_dim, bar_encoder_output_dim):
        super(Generator, self).__init__()

        self.batch_size = batch_size
        self.n_tracks = output_shape[0]
        self.n_bars = output_shape[1]
        self.bar_encoder_output_dim = bar_encoder_output_dim

        self.melody_encoder = MelodyEncoder(melody_output_dim)
        self.chord_temporal_network = TemporalNetwork(z_dim, self.n_bars)
        self.track_temporal_networks = [TemporalNetwork(z_dim, self.n_bars) for _ in range(self.n_tracks - 1)]
        #self.bar_encoder = BarEncoder(bar_encoder_output_dim)
        #self.track_bar_generators = [BarGenerator(melody_output_dim + z_dim * 4 + bar_encoder_output_dim) for _ in range(self.n_tracks - 1)]
        self.track_bar_generators = [BarGenerator(melody_output_dim + z_dim * 4 ) for _ in range(self.n_tracks - 1)]


    def forward(self, chord_noise, style_noise, tracks_noise, groove_noise, conditionnal_melody):
        """
          Chord noise: (batch, 1 bar, z_dim, 1)
          Style noise: (batch, 1 bar, z_dim, 1)
          Tracks noise: (batch, N-1 tracks, 1 bar, z_dim, 1)
          Groove noise: (batch, N-1 tracks, 1 bar, z_dim, 1)
          Conditional melody: (batch, 1 track, N bars, 96, 128)
        """
        batch_size = chord_noise.shape[0]

        ####################### Build inputs #######################
        # Encoded melody is (batch, N_bars, melody_output_dim, 1)
        encoded_melody = self.melody_encoder(conditionnal_melody).unsqueeze(-1)

        # Chord noise extended is (batch, N bars, z_dim, 1)
        chord_extended = self.chord_temporal_network(chord_noise)

        # Tracks noise extended is (N tracks, (batch, N bars, z_dim, 1))
        tracks_noise_extended = []
        for track in range(self.n_tracks - 1):
            track_noise_extended = self.track_temporal_networks[track](tracks_noise[:,track].unsqueeze(1))
            tracks_noise_extended.append(track_noise_extended)

        ####################### Concatenate #######################
        # Track bars will be dimension (N_bars, N_tracks-1, (batch, 1, melody_output_dim + z_dim * 4, 1))
        track_bars = []

        for bar in range(self.n_bars):
            tracks = []
            for track in range(self.n_tracks - 1):
                tracks.append(torch.cat((
                    style_noise.squeeze(1), #(batch, z_dim, 1)
                    groove_noise[:, track, :, :, :].squeeze(1), #(batch, z_dim, 1)
                    chord_extended[:, :, bar, :], #(batch, z_dim, 1)
                    tracks_noise_extended[track][:, :, bar, :] #(batch, z_dim, 1)
                ), dim=1).unsqueeze(1) #(batch ,1 bar, z_dim*4, 1)
                             )
                track_bars.append(tracks)

        ####################### Generate bars #######################
        #previous_bar = torch.zeros(batch_size, 1, self.bar_encoder_output_dim, 1)
        # Store whole bars
        song = []
        for bar in range(self.n_bars):
            generated_tracks_bar = []
            for track in range(self.n_tracks-1):
                # Concat previous bar: (batch, 1 bar, melody_output_dim + z_dim * 4 + bar_encoder_output_dim, 1)
                #track_bar = torch.cat((encoded_melody[:,bar].unsqueeze(1),track_bars[bar][track], previous_bar), dim=2)
                track_bar = torch.cat((encoded_melody[:,bar].unsqueeze(1),track_bars[bar][track]), dim=2)
                # Generated track bar: (batch, 1 bar, 96, 128)
                generated_track_bar = self.track_bar_generators[track](track_bar)
                # Append unsqueezed track bar: (batch, 1 track, 1 bar, 96, 128)
                generated_tracks_bar.append(generated_track_bar.unsqueeze(1))
            # Concat track: (batch, N-1 tracks, 1 bar, 96, 128)
            whole_bar = torch.cat(generated_tracks_bar, dim=1)
            # Add conditional melody: (batch, N tracks, 1 bar, 96, 128)
            whole_song_bar = torch.cat((whole_bar, conditionnal_melody[:,:,bar].unsqueeze(2)), dim=1)
            # Unsqueeze encoded previous bar (batch, bar_encoder_output_dim) -> (batch, 1, bar_encoder_output_dim, 1)
            #previous_bar = self.bar_encoder(whole_song_bar).unsqueeze(1).unsqueeze(-1)
            # Store whole bar
            song.append(whole_song_bar)

        # Concat bars: (batch, N tracks, 1 bar, 96, 128) -> (batch, N tracks, N bars, 96, 128)
        song = torch.cat(song, dim=2)

        return song

