import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch import Tensor

from checkered_layers import CheckeredConv2d

import numpy as np
from os.path import isfile
import cPickle as pickle

class CheckeredCNN(nn.Module):
        # A tiny CCNN with 93,833 parameters. With minor data augmentations, achieves test errors competitive
        # with Capsule Networks (8.2 million parameters)
        def __init__(self):
            super(CheckeredCNN, self).__init__()
            self.init_conv = CheckeredConv2d(1, 8, kernel_size=64, stride=1 , padding=1)
            self.conv1 = CheckeredConv2d(8, 8, kernel_size=32, stride=1, padding=1)
            self.conv2 = CheckeredConv2d(8, 16, kernel_size=32 ,stride=1, padding=1)
            self.conv3 = CheckeredConv2d(16, 16, kernel_size=16,  stride=1, padding=1)
            self.conv4 = CheckeredConv2d(16, 32, kernel_size=16,  stride=1, padding=1)
            self.conv5 = CheckeredConv2d(32, 32, kernel_size=16,  depth=2, padding=1)
            self.bn1 = nn.BatchNorm3d(8)
            self.bn2 = nn.BatchNorm3d(8)
            self.bn3 = nn.BatchNorm3d(16)
            self.bn4 = nn.BatchNorm3d(16)
            self.bn5 = nn.BatchNorm3d(32)
            self.bn6 = nn.BatchNorm3d(32)

            self.pool1 = nn.MaxPool3d((1,1,8))
            self.pool2 = nn.MaxPool3d((1,1,6))
            self.pool3 = nn.MaxPool3d((1,1,4))
            self.drop = nn.Dropout3d(0.5)
            self.fc1 = nn.Linear(16*1*9*161,64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, 2)

        def forward(self, x):
            bs = x.size(0)
            x = x.view(-1, 1, 1, 1, 8192)
            #print(x.size())
            x = self.bn1(F.elu(self.init_conv(x)))
            x = self.bn2(F.elu(self.pool1(self.conv1(x))))
            x = self.bn3(F.elu(self.conv2(x)))
            x = self.bn4(F.elu(self.pool2(self.conv3(x))))
            #x = self.bn5(F.elu(self.conv4(x)))
            #x = self.bn6(F.elu(self.pool3(self.conv5(x))))
            #x = x.view(bs, 64,-1 )A

            #print(x.size())
            #x = x.view(-1, 16*8*9*18)
            x = x.view(-1, 16*1*9*161)
            #x = x.view(-1, 16*1*9*161)
            #x = x.view(-1, 32*3*13*3)
            x = F.elu(self.fc1(x))
            x = self.drop(x)
            x = F.elu(self.fc2(x))
            x = self.drop(x)
            x = self.fc3(x)
            logits = F.softmax(x)
            return logits

def concatenate_datasets(training_dataset, val_dataset, test_dataset, Nts, Nval = 10000, Ntot = 10):
    """
    shorten and concatenate data
    :param initial_dataset: first dataset in the set
    :param Nts: total number of images/time series
    :param Ntot: total number of available datasets
    :return:
    """

    # get core name of dataset without number (_0)
    name = training_dataset.split('_0')[0]
    val_name = val_dataset.split('_0')[0]
    test_name = test_dataset.split('_0')[0]
    print('Using training data for: {0}'.format(name))
    print('Using validation data for: {0}'.format(val_name))
    print('Using test data for: {0}'.format(test_name))

    # load in dataset 0
    with open(training_dataset, 'rb') as rfp:
        base_train_set = pickle.load(rfp)

    with open(val_dataset, 'rb') as rfp:
        base_valid_set = pickle.load(rfp)

    with open(test_dataset, 'rb') as rfp:
        base_test_set = pickle.load(rfp)

    # size of data sets
    size = len(base_train_set[0])
    val_size = len(base_valid_set[0])
    # number of datasets -  depends on Nts
    Nds = np.floor(Nts / float(size))
    # check there are sufficient datasets
    if not Nds <= Ntot:
        print('Error: Insufficient datasets for number of time series')
        exit(0)

    # start with training set
    # if more than the initial data set is needed
    if Nds > 1:
        # how many images/time series needed
        need = Nts - size

        # loop over enough files to reach total number of time series
        for fn in range(1,int(Nds)):
            # load in dataset
            dataset = '{0}_{1}.sav'.format(name,fn)
            with open(dataset, 'rb') as rfp:
                train_set = pickle.load(rfp)
            # check if this set needs truncating
            if need > size:
                cut = size
            else:
                cut = need

            # empty arrays to populate
            aug_train_set = np.zeros(2, dtype = np.ndarray)
            # concatenate the arrays
            for i in range(2):
                aug_train_set[i] = np.concatenate((base_train_set[i], train_set[i][:cut]), axis=0)
            # copy as base set for next loop
            base_train_set = aug_train_set


            need -= size


    else:
        # return truncated version of the initial data set
        aug_train_set = np.zeros(2, dtype=np.ndarray)

        for i in range(2):
            aug_train_set[i] = base_train_set[i][:Nts]

        base_train_set = aug_train_set

    # validation/testing fixed at 10K
    Nds_val = np.floor(Nval / float(val_size))
    # check there are sufficient datasets
    if not Nds_val <= Ntot:
        print('Error: Insufficient datasets for number of time series')
        exit(0)

    if Nds_val > 1:
        # how many images/time series needed
        need = Nval - val_size

        # loop over enough files to reach total number of time series
        for fn in range(1,int(Nds_val)):
            # load in dataset
            val_dataset = '{0}_{1}.sav'.format(val_name,fn)
            test_dataset = '{0}_{1}.sav'.format(test_name,fn)
            with open(val_dataset, 'rb') as rfp:
                valid_set = pickle.load(rfp)
            with open(test_dataset, 'rb') as rfp:
                test_set = pickle.load(rfp)
            # check if this set needs truncating
            if need > val_size:
                cut = val_size
            else:
                cut = need

            # empty arrays to populate
            aug_valid_set = np.zeros(2, dtype = np.ndarray)
            aug_test_set = np.zeros(2, dtype=np.ndarray)
            # concatenate the arrays
            for i in range(2):
                aug_valid_set[i] = np.concatenate((base_valid_set[i], valid_set[i][:cut]), axis=0)
                aug_test_set[i] = np.concatenate((base_test_set[i], test_set[i][:cut]), axis=0)

            # copy as base set for next loop
            base_valid_set = aug_valid_set
            base_test_set = aug_test_set

            need -= val_size


    else:
        # return truncated version of the initial data set
        aug_valid_set = np.zeros(2, dtype=np.ndarray)
        aug_test_set = np.zeros(2, dtype=np.ndarray)

        for i in range(2):
            aug_valid_set[i] = base_valid_set[i][:Nval]
            aug_test_set[i] = base_test_set[i][:Nval]

        base_valid_set = aug_valid_set
        base_test_set = aug_test_set

    return base_train_set, base_valid_set, base_test_set


def truncate_dataset(dataset, start, length):
    """

    :param dataset:
    :param start:
    :param end:
    :return:
    """
    print('    length of data prior to truncating: {0}'.format(dataset[0].shape))
    print('    truncating data between {0} and {1}'.format(start, start+length))
    # shape of truncated dataset
    new_shape = (dataset[0].shape[0],1,length)
    # array to populate
    #truncated_data = np.empty(new_shape, dtype=np.ndarray)
    # loop over data and truncate
    #for i,ts in enumerate(dataset[0]):
    #    truncated_data[i] = ts[0,start:(start+length)].reshape(1,length)

    dataset[0] = dataset[0][:,:,start:(start+length)]
    print('    length of truncated data: {}'.format(dataset[0].shape))
    return dataset



def load_data():
    """
    Load the data set
    :param dataset: the path to the data set (string)
    :param Nts: total number of time series for training
    :return: tuple of theano data set
    """

    # Location and name of training/validation/test sets:
    # set for use on deimos
    datapath='/home/chrism/deepdata_bbh'
    training_dataset='/home/chrism/deepdata_bbh/BBH_training_1s_8192Hz_10Ksamp_25n_iSNR8_Hdet_astromass_1seed_ts_0.sav'
    val_dataset='/home/chrism/deepdata_bbh/BBH_validation_1s_8192Hz_10Ksamp_25n_iSNR8_Hdet_astromass_1seed_ts_0.sav'
    test_dataset='/home/chrism/deepdata_bbh/BBH_testing_1s_8192Hz_10Ksamp_25n_iSNR8_Hdet_astromass_1seed_ts_0.sav'

    Ntot = 10
    Nts = int(1e4)
    Nval = int(1e3)

    train_set, valid_set, test_set = concatenate_datasets(
        training_dataset, val_dataset, test_dataset,
        Nts,Nval=Nval,Ntot=Ntot)

    start = 4096
    length = 8192
    print('Truncating training set')
    train_set = truncate_dataset(train_set,start, length)
    print('Truncating validation set')
    valid_set = truncate_dataset(valid_set,start, length)
    print('Truncating test set')
    test_set = truncate_dataset(test_set, start, length)

    Ntrain = train_set[0].shape[0]
    xshape = train_set[0].shape[1]
    yshape = train_set[0].shape[2]
    channels = 1

    rescale = False

    if rescale:
        print('Rescaling data')
        for i in range(Ntrain):
            train_set[0][i] = preprocessing.normalize(train_set[0][i])

        for i in range(Nval):
            valid_set[0][i] = preprocessing.normalize(valid_set[0][i])
            test_set[0][i] = preprocessing.normalize(test_set[0][i])

    def to_categorical(y, num_classes):
        return np.eye(num_classes, dtype='uint8')[y]

    x_train = (train_set[0].reshape(Ntrain, channels,1, xshape, yshape))
    y_train = train_set[1]#to_categorical(train_set[1], num_classes=2)
    x_val = (valid_set[0].reshape(valid_set[0].shape[0], channels,1, xshape, yshape))
    y_val = valid_set[1]#to_categorical(valid_set[1], num_classes=2)
    x_test = (test_set[0].reshape(test_set[0].shape[0], channels,1, xshape, yshape))
    y_test = test_set[1]#to_categorical(test_set[1], num_classes=2)


    print(y_train[0:10])
    print('Traning set dimensions: {0}'.format(x_train.shape))
    print('Validation set dimensions: {0}'.format(x_val.shape))
    print('Test set dimensions: {0}'.format(x_test.shape))


    return Tensor(x_train), Tensor(y_train).long(), Tensor(x_val), Tensor(y_val).long(), Tensor(x_test), Tensor(y_test).long()


def run(n_epochs=10):
    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            data.requires_grad_()
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 500 == 0:
                print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                    epoch, n_epochs, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
    def test():
        model.eval()
        test_loss = 0
        correct = 0
        results = []
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            results.append((output.data).cpu().numpy())
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        acc = 100. * float(correct) / len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(test_loader.dataset), acc))

        np.save('./results5.npy', results)

        return acc

    def val():
        model.eval()
        val_loss = 0
        correct = 0
        for data, target in val_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            val_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        acc = 100. * float(correct) / len(val_loader.dataset)
        val_loss /= len(val_loader.dataset)
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            val_loss, correct, len(val_loader.dataset), acc))
        return acc

    model = CheckeredCNN().cuda()
    print("\nParameter count: {}\n".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    ))
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)

    # load data
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)

    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)

    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=20, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=20, shuffle=False)

    if not isfile('./targets.npy'):
        np.save('./targets.npy', y_test)

    totrain = True
    totest = True

    if totrain:
        best_acc = 0
        for epoch in range(1, n_epochs + 1):
            scheduler.step()
            train(epoch)
            with torch.no_grad():
                this_acc = val()
                if this_acc > best_acc:
                    torch.save(model.state_dict(), './net.pth')
                best_acc = max(this_acc, best_acc)
            print("Best val accuracy: {:.2f}%\n".format(best_acc))

    if totest:
        model.load_state_dict(torch.load('./net.pth'))
        print("Test accuracy: {:.1f}%\n").format(test())


if __name__ == "__main__":
    """
    Trains our example CCNN on MNIST, which achieves very high accuracy with just 93,833 parameters.
    Args:
        --data_path (string) - path to the directory with your MNIST dataset (will automatically download if it doesn't exist)

    To train the model:
    python demo_mnist.py --data_path <path_to_data_dir>

    Other args:
        --n_epochs (int) - number of epochs for training (default 100)
    """
    fire.Fire(run)
