import torch
import numpy as np
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import argparse
import pickle
import random
def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))

class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, args):
        self.device = device
        self.P = args.window
        self.h = args.horizon
        file_name='COVID.pkl'
        fin = open(file_name, 'rb')

        self.rawdat = pickle.load(fin)
        # self.rawdat = np.loadtxt(fin, delimiter=',' , skiprows=1)
        #self.rawdat = numpy.genfromtxt(fin, delimiter='\t')
        # self._normalized(args.normalize)
        self.perc = args.perc
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.d, self.m = self.dat.shape
        self.scale = np.ones(self.m)
        self.new_normalized()
        self.dat_new = self.dat.reshape((self.n, self.m*self.d))
        self.m_old = self.m
        self.m = self.m * self.d
        self.dat_perc, self.missing_mask = self.generate_miss_3D(self.dat_new, self.perc)

        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.as_tensor(self.scale, device=device, dtype=torch.float)
        # tmp = self.test[2] * self.scale.expand(self.test[2].size(0) * self., self.m)

        self.scale = Variable(self.scale)
        fin.close()

        # self.rse = normal_std(tmp)
        # self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

    def MinMax3D(self, data):  # Input:Samp*var*station,Output:samp*var*station(after normalization)
        from sklearn.preprocessing import MinMaxScaler
        import numpy as np
        data_after = []
        scaler = MinMaxScaler(feature_range=(-1, 1))
        for i in range(data.shape[2]):
            data_after.append(scaler.fit_transform(data[:, :, i]))
        data_after = np.array(data_after)
        data_after = data_after.transpose(1, 2, 0)  # (503,3,7)
        return data_after

    def generate_miss_3D(self, dat_new, perc):
        num = dat_new.shape[0] * dat_new.shape[1]
        list_1 = [ 1 for _ in range(round( num* (1-perc) ) ) ]
        list_nan = [ np.nan for _  in range(round(num * perc)) ]
        list_1.extend(list_nan)
        random.shuffle(list_1)
        list_1 = np.asarray(list_1).reshape(dat_new.shape[0], dat_new.shape[1])
        dat_perc = dat_new * list_1
        dat_perc[np.isnan(dat_perc)] = 0
        # dat_missing = np.array(dat_perc)
        # dat_perc = knn_imputer.fit_transform(dat_missing)
        missing_mask = []
        for i in range(dat_new.shape[1]):
            missing_mask.append(np.isnan(np.array(dat_perc[:, i])))
        missing_mask = np.array(missing_mask).reshape(dat_new.shape[0], dat_new.shape[1])
        return dat_perc, missing_mask

    def new_normalized(self):
        for i in range(self.m):
            self.scale[i] = np.max(np.abs(self.rawdat[:, :, i]))
            self.dat[:, :, i] = self.rawdat[:, :, i] / np.max(np.abs(self.rawdat[:, :, i]))

    def _split(self, train, valid, test):
        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, test)
        self.train = self._batchify(train_set)
        self.valid = self._batchify(valid_set)
        self.test = self._batchify(test_set)

    def _batchify(self, idx_set):
        n = len(idx_set)

        X = torch.zeros((n, self.P, self.m), device=self.device)
        mask = torch.zeros((n, self.P, self.m)).type(torch.int8).to(device=self.device)
        Y = torch.zeros((n, self.P, self.m), device=self.device)

        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.as_tensor(self.dat_perc[start:end, :], device=self.device)
            mask[i,:,:] = torch.as_tensor(self.missing_mask[start:end, :]).to(device=self.device)
            Y[i, :, :] = torch.as_tensor(self.dat_new[start:end, :], device=self.device)

        return [X, mask, Y]

    def get_batches(self, inputs, masks, targets, batch_size, shuffle=False):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length,device=self.device)
        else:
            index = torch.as_tensor(range(length),device=self.device,dtype=torch.long)
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            m = masks[excerpt]
            Y = targets[excerpt]
            yield Variable(X), Variable(m), Variable(Y)
            start_idx += batch_size


def args_def():
    parser = argparse.ArgumentParser(description='PyTorch Time series')
    parser.add_argument('--data_name', type=str, default="ind", help='name of the data file')
    parser.add_argument('--data', type=str, default="data/multiple_industry.txt", help='location of the data file')
    parser.add_argument('--model', type=str, default='grin', help='')
    parser.add_argument('--window', type=int, default=50, help='window size')
    parser.add_argument('--horizon', type=int, default=2)

    parser.add_argument('--clip', type=float, default=10., help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=24, metavar='N', help='batch size')
    parser.add_argument('--dropout', type=float, default=0.001, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=40, help='random seed')
    parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
    parser.add_argument('--save', type=str, default='save/model.pt', help='path to save the final model')

    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--skip', type=int, default=14)
    parser.add_argument('--hidSkip', type=int, default=14)
    parser.add_argument('--L1Loss', type=bool, default=False)

    parser.add_argument('--perc', type=float,default=0.3)
    parser.add_argument('--output', type=str, default='tanh', help='iteration')
    parser.add_argument('--d_hidden', type=int, default=4, help='iteration')

    args = parser.parse_args()
    return args
