import glob
import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Formule output width of convolution network (same for height)
# W=(W−K+2P)/S+1 where W=width, K=kernel width, P=padding and S=stride

# Formule output width of transpose convolution network (same for height)
# W=(W−1)*S+K-2P where W=width, K=kernel width, P=padding and S=stride

from HarmonyGAN import HarmonyGAN

# Load Dataset:

def get_dir(path, no_path=False):
    folder = []
    if not no_path:
        for f in (os.listdir(path)):
            if not f.startswith('.'):
                folder.append(f)
    else:
        for f in (os.listdir()):
            if not f.startswith('.'):
                folder.append(f)
    folder.sort()
    return folder

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.items = glob.glob(folder + '/*')

    def __getitem__(self, item):
        songs_np = np.load(self.items[item])
        return torch.tensor(songs_np)

    def __len__(self):
        return len(get_dir(self.folder + "/"))

LPD_path = "lpd_5/lpd_5_cleansed"
size_bar = 96
n_bars_per_tracks = 8
npy_path = "npy_array"

first_index = 0

dataset = CustomDataset(folder=npy_path)


batch_size = 8


# Train Model:

harmony = HarmonyGAN(n_bars=8,batch_size=batch_size)

harmony.train(dataset,epochs=8,save_every_n_epochs=3,d_loops=5,clamp_weights=0.01,lr_D=0.001)

harmony.show_losses()
harmony.save_model()


final_np=np.load('small_dataset.npy').astype(np.float32)
# Order: Drums, Piano, Guitar, Bass, Strings
final_np = final_np[:,:,:,:,[1,2,4,3,0]]
# Order: Piano, Guitar, Strings, Bass, Drums
ds = TensorDataset(torch.from_numpy(final_np))



[reference_song] = ds[15]
reference_song = reference_song.unsqueeze(0).permute(0,4,1,2,3)
melody = reference_song[:,0,:,:,:].unsqueeze(1)


accompaniement = harmony.accompaniement(melody,thresh=0.3)

def tensor_song_to_array(t_song):
    if type(t_song)==torch.Tensor:
        t_song = t_song.data.numpy()
    _,nb_tracks,nb_bars,steps_per_bar,pitches = t_song.shape
    song = t_song.reshape((nb_tracks,nb_bars*steps_per_bar,pitches))
    return song


accompaniement = tensor_song_to_array(accompaniement)
reference_song = tensor_song_to_array(reference_song)


import pypianoroll


def array_to_pypianoroll(array,tempo=60):
    # Order: Piano, Guitar, Strings, Bass, Drums
    programs = [1, # Accoustic Piano
                29, # Electric muted guitar
                49, # Orchestral Strings
                34, # Electric Bass Finger
                118, # DrumSet
               ]
    is_drum = [False,False,False,False,True]
    tracks = []
    for track in range(array.shape[0]):
        tracks.append(pypianoroll.Track(pianoroll=array[track,:,:],
                                        program=programs[track],
                                        is_drum=is_drum[track]))
    return pypianoroll.Multitrack(tracks=tracks,tempo=tempo,beat_resolution=96//4)


accompaniement = array_to_pypianoroll(accompaniement)
reference_song = array_to_pypianoroll(reference_song)

fig,ax=pypianoroll.plot_multitrack(reference_song,track_label='program')
fig.set_size_inches(10,10)
plt.savefig('pianoroll_reference.png')

fig,ax=pypianoroll.plot_multitrack(accompaniement,track_label='program')
fig.set_size_inches(10,10)
plt.savefig('pianoroll_generated.png')

reference_song.write('reference.mid')
accompaniement.write('generated.mid')
