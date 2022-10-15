import math
import torch.optim as optim
import torch
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import time
import timesynth as ts
import pandas as pd
import torch.utils.data as data_utils
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error

percentageArray = [i for i in range(10, 91, 10)]
maskedPercentages = [ i for i in range(0, 101, 10)]
def evaluate(data, X, Mask, Y, model, evaluateL2, evaluateL1,  args):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    for x_batch, m_batch, y_batch in data.get_batches(X, Mask, Y, args.batch_size, False):
        x_batch, m_batch, y_batch = x_batch.unsqueeze(-1), m_batch.unsqueeze(-1), y_batch.unsqueeze(-1)
        output = model(x_batch, m_batch)
        if predict is None:
            predict = output.clone().detach()
            test = y_batch
        else:
            predict = torch.cat((predict, output.clone().detach()))
            test = torch.cat((test, y_batch))

        scale = data.scale.expand(output.size(0) * output.size(1) * data.d, data.m_old)
        scale = scale.reshape(output.size(0) * output.size(1), data.m)
        output, y_batch = output.reshape(output.size(0) * output.size(1), data.m), \
                    y_batch.reshape(output.size(0) * output.size(1), data.m)

        total_loss += float(evaluateL2(output, y_batch).data.item())
        total_loss_l1 += float(evaluateL1(output, y_batch).data.item())

        n_samples += int((output.size(0) * output.size(1)))

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    rmse = math.sqrt(total_loss / n_samples)
    mae = (total_loss_l1 / n_samples)
    # p_tmp = predict.reshape(predict.shape[0], predict.shape[1]*predict.shape[2]*predict.shape[3])
    # Y_tmp = Ytest.reshape(Ytest.shape[0], Ytest.shape[1] * Ytest.shape[2] * Ytest.shape[3])
    # mape_now = mape(p_tmp, Y_tmp)
    mape_now = mean_absolute_percentage_error(predict.flatten(), Ytest.flatten())
    mre_now = metrics.mean_squared_error(predict.flatten(), Ytest.flatten())
    return rmse, mae, mape_now, mre_now

def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))

def mre(y_true, y_pred):
    return np.sum(np.abs(y_pred - y_true))/np.sum(np.abs(y_true))

def train(data, X, Mask, Y, model, criterion, optim, args):
    model.train()
    total_loss = 0
    n_samples = 0

    for x_batch, m_batch, y_batch in data.get_batches(X, Mask, Y, args.batch_size, False):
        x_batch, m_batch, y = x_batch.unsqueeze(-1), m_batch.unsqueeze(-1), y_batch.unsqueeze(-1)
        output = model(x_batch, m_batch)
        scale = data.scale.expand(output.size(0) * output.size(1) * data.d, data.m_old)
        scale = scale.reshape(output.size(0) * output.size(1), data.m)
        output, y = output.reshape(output.size(0) * output.size(1), data.m),\
                    y.reshape(output.size(0) * output.size(1), data.m)
        # loss = criterion(output * scale, y * scale)
        loss = criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optim.step()
        total_loss += loss.data.item()
        n_samples += int(output.size(0) * output.size(1))

    return total_loss / n_samples

def cuda_available():
    use_cuda = torch.cuda.is_available()
    return use_cuda

def makeOptimizer(params, args):
    if args.optim == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, )
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(params, lr=args.lr, )
    elif args.optim == 'adadelta':
        optimizer = optim.Adadelta(params, lr=args.lr, )
    elif args.optim == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, )
    else:
        raise RuntimeError("Invalid optim method: " + args.method)
    return optimizer

