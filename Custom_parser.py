import os
import numpy  as np
import pandas as pd
from pathlib import Path
from   NILM_Dataset     import *
from   Pretrain_Dataset import *

class Custom_Parser:

    def __init__(self, args, stats = None):
        self.dataset_location = args.custom_location
        self.data_location    = Path(args.custom_location)

        self.appliance_names  = args.appliance_names
        self.sampling         = args.sampling
        self.normalize        = args.normalize

        self.house_indicies  = args.house_indicies


        self.cutoff        =  [args.cutoff[appl]    for appl in ['aggregate']+args.appliance_names]
        self.threshold     =  [args.threshold[appl] for appl in args.appliance_names]
        self.min_on        =  [args.min_on[appl]    for appl in args.appliance_names]
        self.min_off       =  [args.min_off[appl]   for appl in args.appliance_names]

        self.val_size      =  args.validation_size
        self.window_size   =  args.window_size
        self.window_stride =  args.window_stride

        
        self.x, self.y     = self.load_data()
        print(self.x.shape)
        print(self.y.shape)
        if self.normalize == 'mean':
            if stats is None:
                self.x_mean = np.mean(self.x)
                self.x_std  = np.std(self.x)
                print("mean", self.x_mean)
                print("std", self.x_std)
            else:
                self.x_mean,self.x_std = stats
            self.x = (self.x - self.x_mean) / self.x_std

        self.status = self.compute_status(self.y)
        print(np.sum(self.status))



    def load_data(self):

        entire_main = []
        entire_appl = []

        for house_idx in self.house_indicies:
            filename  = f'house{house_idx}.npy'
            main_loc= f'{self.data_location}/aggregate/{filename}'
            appl_loc= f'{self.data_location}/{self.appliance_names[0]}/{filename}' 
            
            entire_main.append(np.load(main_loc))
            entire_appl.append(np.load(appl_loc))
        x = np.concatenate(entire_main, axis=0,  dtype=np.float32).squeeze()
        y = np.concatenate(entire_appl, axis=0,  dtype=np.float32).squeeze()
        return x, y





    def compute_status(self,data):
        initial_status = data >= self.threshold[0]
        status_diff    = np.diff(initial_status)
        events_idx     = status_diff.nonzero()

        events_idx  = np.array(events_idx).squeeze()
        events_idx += 1

        if initial_status[0]:
            events_idx = np.insert(events_idx, 0, 0)

        if initial_status[-1]:
            events_idx = np.insert(events_idx, events_idx.size, initial_status.size)

        events_idx     = events_idx.reshape((-1, 2))
        on_events      = events_idx[:, 0].copy()
        off_events     = events_idx[:, 1].copy()
        assert len(on_events) == len(off_events)

        if len(on_events) > 0:
            off_duration = on_events[1:] - off_events[:-1]
            off_duration = np.insert(off_duration, 0, 1000)
            on_events    = on_events[off_duration > self.min_off[0]]
            off_events   = off_events[np.roll(off_duration, -1) > self.min_off[0]]

            on_duration  = off_events - on_events
            on_events    = on_events[on_duration  >= self.min_on[0]]
            off_events   = off_events[on_duration >= self.min_on[0]]
            assert len(on_events) == len(off_events)

        temp_status = data.copy()
        temp_status[:] = 0
        for on, off in zip(on_events, off_events):
            temp_status[on: off] = 1
        status = temp_status

        return status    

    def get_train_datasets(self):
        val_end = int(self.val_size * len(self.x))
        
        val = NILMDataset(self.x[:val_end],
                          self.y[:val_end],
                          self.status[:val_end],
                          self.window_size,
                          self.window_size    #non-overlapping windows
                          )

        train = NILMDataset(self.x[val_end:],
                            self.y[val_end:],
                            self.status[val_end:],
                            self.window_size,
                            self.window_stride
                            )
        return train, val

    def get_pretrain_datasets(self, mask_prob=0.25):
        val_end = int(self.val_size * len(self.x))

        val  = NILMDataset(self.x[:val_end],
                           self.y[:val_end],
                           self.status[:val_end],
                           self.window_size,
                           self.window_size
                          )
        train = Pretrain_Dataset(self.x[val_end:],
                                 self.y[val_end:],
                                 self.status[val_end:],
                                 self.window_size,
                                 self.window_stride,
                                 mask_prob=mask_prob
                                )
        return train, val
