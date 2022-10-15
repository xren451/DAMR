import math
import time

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import pathlib
import torch
import torch.nn as nn
from models import grin
import importlib
import warnings
warnings.filterwarnings("ignore")
import datetime
import yaml
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from utils import *
from utils import args_def
from train_eval import train, evaluate, makeOptimizer

args = args_def()
print(args.model)
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility.
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
# Load data
Data = Data_utility(args.data, 0.8, 0.1, device, args)

exp_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.seed}"
logdir = os.path.join('/Users/macbook/Desktop/PHD/CIKM/Baseline/QY', "COVID19", "SpatialConvOrderK", exp_name)
# save config for logging
pathlib.Path(logdir).mkdir(parents=True)

# loss function
if args.L1Loss:
    criterion = nn.L1Loss(size_average=True)
else:
    criterion = nn.MSELoss(size_average=True)
evaluateL2 = nn.MSELoss(size_average=False)
evaluateL1 = nn.L1Loss(size_average=False)
if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda()
    evaluateL2 = evaluateL2.cuda()

# Select model
model = eval(args.model)
model = model.Model(d_hidden=args.d_hidden)

train_method = train
eval_method = evaluate
Tloss, Validrse = list(), list()
nParams = sum([p.nelement() for p in model.parameters()])
print('number of parameters: %d' % nParams)
model = model.to(device)
best_val = 10000000

optim = makeOptimizer(model.parameters(), args)

try:
    print('Training start')
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train_method(Data, Data.train[0], Data.train[1], Data.train[2], model, criterion, optim, args)
        val_loss, val_rae, val_mape, val_mre = eval_method(Data, Data.valid[0], Data.valid[1], Data.valid[2], model, evaluateL2, evaluateL1, args)
        print('| end of epoch {:3d} | time used: {:5.2f}s | train_loss {:5.4f} | valid rmse {:5.4f} | valid mae {:5.4f} | valid mape {:5.4f} | valid mre {:5.4f}'.
                format( epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_mape, val_mre))
        Tloss.append(train_loss)
        Validrse.append(val_loss)
        if val_loss < best_val:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val = val_loss
        if epoch % 10 == 0:
            test_acc, test_rae, test_mape, test_mre = eval_method(Data, Data.test[0], Data.test[1], Data.test[2], model, evaluateL2, evaluateL1,args)
            print("| test rmse {:5.4f} | test rae {:5.4f} | test mape {:5.4f}  | test mre {:5.4f} ".format(test_acc, test_rae, test_mape, test_mre))

except KeyboardInterrupt:
    print('-' * 100)
    print('Exiting from training early')

def plot_loss(data_loss):
    fig = plt.figure(facecolor='white',figsize=(25,12))
    plt.plot(data_loss, label='True Data', linewidth=4)
    plt.xlabel("Epochs",fontsize=30)
    plt.ylabel("Training loss",fontsize=30)
    plt.yticks(size=24)
    plt.xticks(size=24)

    timestep = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
    filename = "result/train_loss_" + args.model + "_("+ str(nParams) + ")_"+ timestep + ".png"
    plt.savefig(filename)
    plt.show()
    plt.close()

def plot_val_loss(val_loss):
    fig = plt.figure(facecolor='white',figsize=(25,12))
    plt.plot(val_loss, label='True Data')
    plt.xlabel("Epochs",fontsize=30)
    plt.ylabel("Validation rse",fontsize=30)
    plt.yticks(size=24)
    plt.xticks(size=24)

    timestep = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
    filename = "result/validation_rse_" + args.model + "_("+ str(nParams) + ")_"+ timestep + ".png"
    plt.savefig(filename)
    plt.show()
    plt.close()

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

test_acc, test_rae, test_mape, test_mre = evaluate(Data, Data.test[0], Data.test[1], Data.test[2], model, evaluateL2, evaluateL1,args)

# pred = np.array(predict[:,index])
# base = np.array(Data.test[1].data.cpu().numpy()[:,index])
# plot_results(pred, base, Data, index)
plot_loss(Tloss)
plot_val_loss(val_loss)

print('Best model performance：')
print("| test rmse {:5.4f} | test mae {:5.4f} | test mape {:5.4f}  | test mre {:5.4f} ".format(test_acc, test_rae, test_mape, test_mre))
