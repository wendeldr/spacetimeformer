import mne
import numpy as np

eeg1 = mne.io.read_raw_edf("/home/wendeldr/git/spacetimeformer/spacetimeformer/data/edf/FC_OvertNaming.EDF", preload=True)
eeg2 = mne.io.read_raw_edf("/home/wendeldr/git/spacetimeformer/spacetimeformer/data/edf/PC_OvertNaming.EDF", preload=True)

# filter 60 Hz noise and harmonics with zerophase notch filter
eeg1 = eeg1.notch_filter(np.arange(60, 181, 60), fir_design='firwin', picks=eeg1.info['ch_names'][0]).get_data()
eeg2 = eeg2.notch_filter(np.arange(60, 181, 60), fir_design='firwin', picks=eeg2.info['ch_names'][1]).get_data()


a=1