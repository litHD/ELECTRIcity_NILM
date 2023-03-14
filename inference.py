from Electricity_model import ELECTRICITY
from config import update_preprocessing_parameters
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import json

def cutoff_energy(data, cutoff):

    data[data < 5] = 0
    tmp = np.full(shape=(data.shape[0],),fill_value=cutoff)

    data = np.minimum(data,tmp)
    
    return data

def find_nonzero_runs(a):
    if len(a.shape) ==1:
        zeros = np.zeros((1,))
    if len(a.shape) ==2:
        zeros = np.zeros((1,1))
    # Create an array that is 1 where a is nonzero, and pad each end with an extra 0.
    isnonzero = np.concatenate((zeros, (np.asarray(a) != 0).view(np.int8), zeros))
    absdiff = np.abs(np.diff(isnonzero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

#Func that fuse together  near intervals that are close n_timesteps=gap
def zeros_interval(data, intervals, gap):
    i=0

    for i, interval in enumerate(intervals):
            if interval[1]-interval[0]<= gap:
                data[interval[0]:interval[1]] = 0 

def compute_status(data, treshold, min_on):  
    # sourcery skip: inline-immediately-returned-variable
    status = (data >= treshold) * 1
    status = status.reshape(data.shape)
    interval = find_nonzero_runs(status)
    if interval is not None:
        zeros_interval(data, interval, min_on)
    return status

#da guardare



def padding_seqs(in_array, window_size):
    if len(in_array) == window_size:
        return in_array
    try:
        out_array = np.zeros((window_size, in_array.shape[1]))
    except:
        out_array = np.zeros(window_size)

    length = len(in_array)
    out_array[:length] = in_array
    return out_array

def standardize_data(data, mu=0.0, sigma=1.0):
    data = data-mu
    data /= sigma
    return data
def de_standardize_data(data, mu=0.0, sigma=1.0):
    data *= sigma
    data = data+mu
    return data


def acc_precision_recall_f1_score(pred, status):
    assert pred.shape == status.shape

    pred = pred.reshape(-1, pred.shape[-1])
    status = status.reshape(-1, status.shape[-1])
    accs, precisions, recalls, f1_scores = [], [], [], []

    for i in range(status.shape[-1]):
        tn, fp, fn, tp = confusion_matrix(status[:, i], pred[:, i], labels=[
                                          0, 1]).ravel()
        acc = (tn + tp) / (tn + fp + fn + tp)
        precision = tp / np.max((tp + fp, 1e-9))
        recall = tp / np.max((tp + fn, 1e-9))
        f1_score = 2 * (precision * recall) / \
            np.max((precision + recall, 1e-9))

        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    return np.array(accs), np.array(precisions), np.array(recalls), np.array(f1_scores)

def relative_absolute_error(pred, label):
    assert pred.shape == label.shape

    pred = pred.reshape(-1, pred.shape[-1])
    label = label.reshape(-1, label.shape[-1])
    temp = np.full(label.shape, 1e-9)
    relative, absolute, sum_err = [], [], []

    for i in range(label.shape[-1]):
        relative_error = np.mean(np.nan_to_num(np.abs(label[:, i] - pred[:, i]) / np.max(
            (label[:, i], pred[:, i], temp[:, i]), axis=0)))
        absolute_error = np.mean(np.abs(label[:, i] - pred[:, i]))

        relative.append(relative_error)
        absolute.append(absolute_error)

    return np.array(relative), np.array(absolute)

def Wh_estimate(pred, label, f_sampling):
    assert pred.shape == label.shape
    label_p = np.sum(label, axis=0)/(3600*f_sampling)
    pred_p = np.sum(pred, axis=0)/(3600*f_sampling)
    pred_percentage = pred_p/label_p
    return pred_percentage, label_p
    
def log_data_and_images(path,aggregate,pred,labels,f_sampling,appliance,args,show=False):
   
    path = f"{path}_{appliance}_train_{args.trained_on}_test_{args.tested_on}_{args.house_id}"
    if len(pred.shape) == 1:
        pred = np.expand_dims(pred,-1)

    if len(labels.shape) == 1:
        labels = np.expand_dims(labels,-1)

    if len(aggregate.shape) == 2:
        aggregate = np.squeeze(aggregate, axis=-1)

    print("prediction shape", pred.shape)
    print("ground truth shape", labels.shape)
    print("aggregate shape", aggregate.shape)

    status_l = compute_status(labels,args.treshold,args.min_on)
    status_p = compute_status(pred,args.treshold,args.min_on)
    pred[pred < args.treshold] = 0

    assert pred.shape == labels.shape
    assert aggregate.shape[0] == pred.shape[0]
    assert status_l.shape == status_p.shape
        
    acc, precision, recall, f1_score = acc_precision_recall_f1_score(status_p, status_l)
    rel_error, abs_error = relative_absolute_error(pred, labels)
    
    wh_percentage, wh_label = Wh_estimate(pred, labels, f_sampling)
    metrics = { 'appliances': appliance,
                'accuracy': list(acc),
                "precision":list(precision),
                "recall":list(recall),
                "f1_score":list(f1_score),
                "relative_error":list(rel_error),
                "absolute_error":list(abs_error),
                "tot_appl_wh": list(wh_label),
                "predicted_percentage": list(wh_percentage)
                }
    
    print(metrics)
    plt.figure(figsize=(50, 10))
    print(labels.shape)
    plt.title(f'Appliance {appliance}')
    plt.plot(range(aggregate.shape[0]), aggregate[:], label='Aggregate')
    plt.plot(range(aggregate.shape[0]), labels[:, 0], label='Ground truth')
    plt.plot(range(aggregate.shape[0]), pred[:, 0], label='Prediction')
    plt.legend()

    
    if os.path.exists(path):

        if len(os.listdir(path)) == 0:
            pass
        else:
            print("found a non empty directory")    
            return
    else:
        os.makedirs(path)
        
    plt.savefig(os.path.join(path, f"{appliance}.png"))
    if show:
        plt.show()
    with open(os.path.join(path,"metrics.json"), 'w') as outfile:
        json.dump(metrics, outfile)
    with open(os.path.join(path,"parameters.json"), 'w') as outfile:
        a = vars(args)
        json.dump(a, outfile)

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--appliance', type=str, default='microwave')
    parser.add_argument('--trained_on', type=str)
    parser.add_argument('--tested_on', type=str) 
    parser.add_argument('--main_path', type=str)
    parser.add_argument('--appliance_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--output_size', type=int, default=1)
    parser.add_argument('--drop_out', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--cutoff', type=float)
    parser.add_argument('--treshold', type=float)
    parser.add_argument('--window_size', type=int, default=600)
    parser.add_argument('--stride', type=int, help='window_size/stride need to be odd integer number')
    parser.add_argument('--until', type=int)
    parser.add_argument('--min_on', type=int)
    parser.add_argument('--f_sampling', type=float, default=1/6)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--n_layers', type=int, default=2)
    #parser.add_argument('--mean_train', type=float, default=None)
    #parser.add_argument('--std_train', type=float, default=None)
    parser.add_argument('--house_id', type=int)
    args = parser.parse_args()
    args.pretrain =False
    args.mul = int(((args.window_size/args.stride)-1)/2)
    args.dataset_code = args.trained_on

    update_preprocessing_parameters(args)

    args.cutoff = args.cutoff[args.appliance]
    args.min_on = args.min_on[args.appliance]
    args.treshold = args.threshold[args.appliance]
    args.c0 = 0
    args.min_off = 0

    model = ELECTRICITY(args)
    print(model)
    model.to(args.device)
    model.float()
    model.load_state_dict(torch.load(args.model_path))

    
    x = np.load(os.path.join(args.main_path, f'house{args.house_id}.npy'))  
    y = np.load(os.path.join(args.appliance_path, f'house{args.house_id}.npy'))

    mean = np.mean(x)
    std = np.std(x)

    model.eval()
    

    # if args.mean_train is not None and args.std_train is not None:
    #     args.inference_cutoff = (args.cutoff+np.abs(mean-args.mean_train))*(std/args.std_train)
    # else:
    #     args.inference_cutoff = args.cutoff
    energy_res = []
    status_res = []
    x = x[:args.until]
    print(y.shape)
    print(x.shape)
    for i in tqdm(np.arange(0, x.shape[0],args.stride)):

        seqs = padding_seqs(x[i:i+args.window_size], args.window_size)
        seqs = standardize_data(seqs, mean, std)
        seqs = np.reshape(seqs,(1,1,args.window_size))
        seqs = torch.tensor(seqs).to(args.device)
        with torch.no_grad():
            logits = model(seqs.float())
            logits = logits[0].cpu().numpy().squeeze()
            logits_energy = cutoff_energy(logits *args.cutoff, float(args.cutoff))
            logits_energy = logits_energy 
        if i==0:
            energy_res.append(logits_energy[:(args.mul+1)*args.stride])

        else:
            energy_res.append(logits_energy[+args.mul*args.stride:-args.mul*args.stride])

        energy = np.concatenate(energy_res)
        energy[energy<10] = 0        

    log_data_and_images('./logs/', x,energy[:x.shape[0]], y[:x.shape[0]], args.f_sampling, args.appliance, args,show=True)
