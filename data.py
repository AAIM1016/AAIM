import numpy as np
from torch.utils.data import Dataset
import pickle

class DataLoad(Dataset):
    def __init__(self, partition='train', dataset='ADNI',fold=0):
        folder="/home/{}/".format(dataset)
        index_path = folder + "index/{}_{}index_fold_{}.txt".format(dataset,partition,fold+1)
        label_path = folder+"label/{}_label.pkl".format(dataset)
        data_path = folder + 'data/{}_GM_WM.npy'.format(dataset)
        self.index = np.loadtxt(index_path).astype('int')
        self.label = []
        self.data = []
        datas=np.load(data_path)
        with open(label_path, 'rb') as f:
            labels= pickle.load(f)
            for _, j in enumerate(self.index):
                self.label.append(labels[j])
                self.data.append(datas[j])
        self.data = np.stack(self.data)

    def __getitem__(self, item):
        data = self.data[item, :, :]
        label = self.label[item]
        return data, label

    def __len__(self):
        return self.data.shape[0]


