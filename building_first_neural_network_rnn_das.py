# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pickle
import torch
def pickleStore(savethings, filename):
  # Store the checkpoint
    dbfile = open( filename , 'wb' )
    pickle.dump( savethings , dbfile )
    dbfile.close()
    return

def pikleOpen(filename):
  # Open the model
    file_to_read = open( filename , "rb" )
    p = pickle.load( file_to_read )
    return p

def readData(f):
  # Read Data from the system
    return np.genfromtxt(f, delimiter=',', dtype=str)[1:]
    # 將資料根據delimiter切分成一個一個，以dtype的型別存入Numpy的array

def saveModel(net, path):
  # Save the model
    torch.save(net.state_dict(), path)

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, labels, device='gpu'):
        'Initialization'
        self.data = data.to(device)#轉換為指定是設備的tensor
        self.labels = labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.data[index]
        y = self.labels[index]
        return X, y, index

from sklearn.preprocessing import MinMaxScaler

def preprocess(data, flip=True):
    date   = data[:, 0]
    open   = data[:, 1]
    high   = data[:, 2]
    low    = data[:, 3]
    close  = data[:, 4]
    volume = data[:, 5]
    # prices = np.array([[d[1], d[4]] for d in data], dtype=np.float64)
    # 取open和close
    # prices = prices.ravel()
    # 將二維數組轉換為一維數組
    prices = np.array([close for date, open, high, low, close, volume in data]).astype(np.float64)
    # 只取close
    if flip:
        prices = np.flip(prices)
    print("price:",prices)
    print(prices.shape)
    # 正規化
    scaler = MinMaxScaler(feature_range=(-1, 1))
    prices = scaler.fit_transform(prices.reshape(-1, 1))
    return prices,scaler


#分為 testing data and validate data
def train_test_split(data, percentage=0.85):
    train_size  = int(len(data) * percentage)
    train, test = data[:train_size], data[train_size:]
    # print("train:",train)
    # print("test:",test)
    return train, test
# open + close
# def transform_dataset(dataset, look_back=10):
#     # N days as training sample
#     dataX = [dataset[i:(i + look_back)]
#             for i in range(0,len(dataset)-look_back-1,2)]
#     # 1 day as groundtruth
#     dataY = [dataset[i + look_back+1]
#             for i in range(0,len(dataset)-look_back-1,2)]
#     return (torch.tensor(np.array(dataX), dtype=torch.float32).squeeze(),
#             torch.tensor(np.array(dataY), dtype=torch.float32).squeeze())
# close
def transform_dataset(dataset, look_back=5):
    # N days as training sample
    dataX = [dataset[i:(i + look_back)]
            for i in range(0,len(dataset)-look_back-1,1)]
    # 1 day as groundtruth
    dataY = [dataset[i + look_back+1]
            for i in range(0,len(dataset)-look_back-1,1)]
    return (torch.tensor(np.array(dataX), dtype=torch.float32).squeeze(),
            torch.tensor(np.array(dataY), dtype=torch.float32).squeeze())

import torch.nn as nn
# 當自定義一個神經網絡模型時，需要繼承 nn.Module 類別，並實現 __init__() 和 forward() 這兩個方法
class LSTMPredictor(nn.Module):
# 定義網絡中要用到的層
    def __init__(self, look_back, num_layers=2, dropout=0.5, bidirectional=True):
        super(LSTMPredictor, self).__init__()

        # Nerual Layers LSTM - Long Short Term Memory
        self.rnn   = nn.LSTM(look_back, 32, num_layers, dropout=dropout, bidirectional=True)
        # LSTM層
        # Batch Normalization layer
        self.batch_norm = nn.BatchNorm1d(32 * (2 if bidirectional else 1))
        # Layer normalize
        self.layer_norm = nn.LayerNorm(32 * (2 if bidirectional else 1))
        # 全連接層
        self.ly_a = nn.Linear(32 * (2 if bidirectional else 1), 16)
        # self.ly_a  = nn.Linear(look_back, 16)
        # self.relu  = nn.ReLU()
        self.relu = nn.Tanh()
        # 激活函數
        # self.relu = nn.Sigmoid()
        self.reg   = nn.Linear(16, 1)
        # 回歸層
        self.num_layers = num_layers
    def predict(self, input,scaler):
        with torch.no_grad():
            output = self.forward(input).item()
            # use the inverse_transform method to reverse the normalization
            output = scaler.inverse_transform([[output]])[0, 0]
            return output
            # 取得標量（純量）
# 定義網絡的前向傳播過程，也就是將輸入映射成輸出的過程
    def forward(self, input):
        # print(input.shape)
        r_out, (h_n, h_c) = self.rnn(input.unsqueeze(1), None)
        # 增加一個維度
        # print("rout",r_out.shape)
        # Apply batch normalization after the LSTM layer
        r_out = r_out.squeeze(1)
        # r_out = self.batch_norm(r_out)
        # Apply layer normalization after the LSTM layer
        # r_out = self.layer_norm(r_out)

        logits = self.reg(self.relu(self.ly_a(r_out)))
        # 去掉一個維度
        # logits = self.reg(self.relu(self.ly_a(input)))

        return logits

import numpy as np

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
def trainer(model, train_loader, optimizer, criterion,epoch,batch_size):
    model.train()
    train_loss = 0.0
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_x, batch_y, data_index = data
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output.cpu(), batch_y.unsqueeze(1).cpu())
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)
        running_loss += loss.item()
        # train_loss /= len(train_loader.dataset)
        if i % 10 == 9:  # print every batch_size mini-batches
            print('[%d, %5d] loss: %.6f' % (epoch + 1, i + 1, train_loss / batch_size))
            running_loss = 0.0
    return train_loss
def validater(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader,0):
            batch_x, batch_y, data_index =data
            output = model(batch_x)
            loss = criterion(output.cpu(), batch_y.unsqueeze(1).cpu())
            val_loss += loss.item() * batch_x.size(0)
            # val_loss /= len(val_loader.dataset)
    return val_loss

def loadModel(path,look_back):
    model = LSTMPredictor(look_back)
    model.load_state_dict(torch.load(path))
    return model

import pandas as pd
import os
def run_kfold(model, dataset, testloader,n_splits=5, n_epochs=10, batch_size=64, lr=0.001, shuffle=True,look_back=10):
    kfold = KFold(n_splits=n_splits, shuffle=shuffle)
    scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print("Fold {}/{}:".format(fold_idx+1, n_splits))
        # Split dataset into training and validation sets
        train_data = dataset[train_idx]
        val_data = dataset[val_idx]
        trainX, trainY = transform_dataset(train_data,look_back=look_back)
        valX, valY = transform_dataset(val_data, look_back=look_back)
        train_set = Dataset(trainX, trainY, device)
        val_set = Dataset(valX,valY,device)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False,num_workers=0)

        # Initialize optimizer and loss function
        # optimizer = optim.SGD(net.parameters(), lr=lr)
        # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        optimizer = optim.Adam(net.parameters(), lr=lr) # you can tweak the lr and see if it affects anything
        # AMSGrad
        # optimizer = optim.Adam(net.parameters(), lr=lr,amsgrad=True) # you can tweak the lr and see if it affects anything
        criterion = nn.MSELoss()
        x_list = []
        y_list = []
        # Train and validate the model for n_epochs
        for epoch in range(n_epochs):
            train_loss = trainer(model, train_loader, optimizer, criterion,epoch,batch_size)
            val_loss = validater(model, val_loader, criterion)
            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)
            x_list.append(train_loss)
            y_list.append(val_loss)
            print("Epoch {}/{} - Train loss: {:.6f}, Validation loss: {:.6f}".format(
                epoch + 1, n_epochs, train_loss, val_loss))

        test = tester(model, criterion, testloader)
        print('Test Result: ', test)

        random_list = [126, 124, 124, 122.5, 121]
        predict = model.predict(torch.tensor([random_list], dtype=torch.float32).to(device),scaler)
        print('Predicted Result', predict)

        y_true = valY.detach().cpu().numpy()
        y_pred = model(valX).detach().cpu().numpy()

        r2 = r2_score(y_true, y_pred)
        scores.append(r2)
        print("R^2 score:", r2)

        plt.figure()
        plt.plot(range(len(x_list)), x_list, "r", label='training loss')
        plt.plot(range(len(y_list)), y_list, "b", label="validation los")
        plt.xlim(0,n_epochs)
        plt.ylim(0,0.05)
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        path = "lk{}e{}b{}lr{}h{}fid{}".format(look_back,n_epochs,batch_size,lr,model.num_layers,fold_idx)
        plt.savefig(path+".png")
        # 儲存結果到字典中
        result_dict = {
            'id':fold_idx,
            'epoch': n_epochs,
            'look_back': look_back,
            'batch_size': batch_size,
            'learning rate': lr,
            'number_layers': model.num_layers,
            'training_loss': x_list[-1],
            'validation_loss': y_list[-1],
            'r2_score': r2,
            'test_result': test,
            'predicted_result': predict.item()
        }

        # 將字典轉換成 DataFrame 對象
        result_df = pd.DataFrame([result_dict])
        # filename = 'result.csv'
        # filename = 'epoch.csv'
        # filename = 'batch.csv'
        # filename = 'lr.csv'
        # filename = 'hl.csv'
        # filename = 'optimizer.csv'
        # filename = 'lnorm.csv'
        # filename = 'bnorm.csv'
        # filename = 'af.csv'
        # filename='lk.csv'
        # filename='pr.csv'
        # filename = 'datasetC.csv'
        filename = 'best.csv'
        # 將 DataFrame 寫入到 CSV 文件中
        result_df.to_csv(filename,float_format="%.6f", index=False, mode='a', header=not os.path.exists(filename))
        # plt.show()
        ## Save model
        saveModel(model, path+"last3"+".pt")
        # Evaluate the model on the validation set and store the score
    mean_score = sum(scores) / n_splits
    print(f"Mean score: {mean_score}")

    print('Finished Training')

def trainer_(net, criterion, optimizer, trainloader, devloader, epoch_n, path="./checkpoint/save.pt"):
    x_list = []
    y_list = []
    for epoch in range(epoch_n): # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        train_loss = 0.0
        valid_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, data_index = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            # outputs = torch.tensor(outputs,dtype=torch.long)
            # print(type(outputs))
            # outputs = net(inputs)
            loss = criterion(outputs.cpu(), labels.unsqueeze(1).cpu())
            train_loss += loss.item()*inputs.shape[0]
            # 計算每個epoch的loss總和
            loss.backward()
            # backpropagation
            optimizer.step()
            # 更新梯度
            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

        ######################    
        # validate the model #
        ######################
        net.eval()
        for i, data in enumerate(devloader, 0):
            # print("vaildate:i,data",i,data)
            # move tensors to GPU if CUDA is available
            inputs, labels, data_index = data
            # print("input:",inputs.shape,"label:",labels.shape,"index:",data_index.shape)
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = net(inputs)
            # calculate the batch loss
            loss = criterion(outputs.cpu(), labels.cpu())
            # update average validation loss 
            valid_loss += loss.item()*inputs.shape[0]
        
        # calculate average losses
        train_loss = train_loss/len(trainloader.dataset)
        valid_loss = valid_loss/len(devloader.dataset)
        x_list.append(train_loss)
        y_list.append(valid_loss)
        # print training/validation statistics 
        print("epoch",epoch,'\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(train_loss, valid_loss))
        
    print('Finished Training')
    plt.plot(range(len(x_list)), x_list, "r", label='training loss')
    plt.plot(range(len(y_list)), y_list, "b", label="validation los")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
    ## Save model
    saveModel(net, path)

def tester(net, criterion, testloader):
    loss = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels, data_index = data
            outputs = net(inputs)
            loss += criterion(outputs.cpu(), labels.unsqueeze(1).cpu())
    return loss.item()

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data.dataloader import default_collate
"""## Get Device for Training
We want to be able to train our model on a hardware accelerator like the GPU,
if it is available. Let's check to see if
[torch.cuda](https://pytorch.org/docs/stable/notes/cuda.html) is available, else we
continue to use the GPU.
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
"""# Read Data"""
data = readData("0050_history.csv")
print('Num of samples:', len(data))
print('type:',type(data))
print(data.shape)
print(data)
"""# Preprocess"""
prices,scaler = preprocess(data)
# Divide trainset and test set
# percentage = 0.7
percentage = 0.85
# percentage = 0.9
train, test = train_test_split(prices,percentage)
# print("train:",train.shape)
# print("test",test.shape)
# Set the N(look_back)=5 because from the five day stock, we are predicting the next day
look_back = 5
# trainX, trainY = transform_dataset(train, look_back)
# print("trx",trainX.shape,"try",trainY.shape)
testX, testY = transform_dataset(test, look_back)
# print("tsx",testX.shape,"tsy",testY.shape)
# Get dataset
# trainset = Dataset(trainX, trainY, device)
testset  = Dataset(testX, testY, device)
# print("trset",trainset.data.shape,trainset.labels.shape)
# print("tsset",testset.data.shape,trainset.labels.shape)
# Get dataloader
batch_size = 200
lr = 0.001
num_layers = 2
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers should set 1 if put data on CUDA
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
"""# Model Initialize"""

net = LSTMPredictor(look_back,num_layers=num_layers)
net.to(device)

"""# Loss Function"""

criterion = nn.MSELoss() # Feel free to use any other loss
# L2
# criterion = nn.CrossEntropyLoss()
# criterion = nn.L1Loss()
# MAE

"""I used Mean Square Error Loss function. You can tweak with other if you want.
https://pytorch.org/docs/stable/nn.html#loss-functions

Or, you can create your own loss function 
https://towardsdatascience.com/how-to-create-a-custom-loss-function-keras-3a89156ec69b

# Optimizer
"""
# optimizer = optim.SGD(net.parameters(), lr=lr)
# optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=lr) # you can tweak the lr and see if it affects anything
# AMSGrad
# optimizer = optim.Adam(net.parameters(), lr=lr,amsgrad=True) # you can tweak the lr and see if it affects anything
"""# Training

You can change the epoch here.
But also make sure it doesn't overfit!
"""
# Training
# checkpoint = "./checkpoint/save.pt"
# if not os.path.isfile(checkpoint):
#   os.makedirs("./checkpoint")
#   # trainer(net, criterion, optimizer, trainloader, testloader, epoch_n=1000, path=checkpoint)
#   run_kfold(net,prices,n_splits=5,n_epochs=1000,batch_size=batch_size,lr=0.001,look_back=look_back)
# else:
#   net.load_state_dict(torch.load(checkpoint))
# run_kfold(net,train,testloader=testloader,n_splits=3,n_epochs=200,batch_size=batch_size,lr=lr,look_back=look_back)


"""# Test the Data
Now let's compare with the test data
"""
# test = tester(net, criterion, testloader)
# Show the difference between predict and groundtruth (loss)
# print('Test Result: ', test)

"""# Predict
Our model is now ready! 

Let's try with some sample data!
Suppose the closing data, 
* Monday = 126
* Tuesday = 124
* Wednesday = 124
* Thursday = 122.5
* Friday = 121

So, the model is predicting the next day given the input of 5 day closing values
"""
import random
# random_list = [random.randint(120, 126) for i in range(5)]
net = LSTMPredictor(look_back,num_layers)
net = loadModel("/Users/lichengyu/Desktop/three_grade2/deep_learning/lk5e200b200lr0.001h2fid1last2.pt",look_back)
random_list = [184.75521272567516,184.98940274667302,186.59876219047715,180.9735626665538,184.34884481977758
]
predict = net.predict(torch.tensor([random_list], dtype=torch.float32).to(device),scaler)
print('Predicted Result', predict)

"""What can you do to improve the performance?

* Increase the epoch
* You can use multiple input data (Maybe closing data and volume has a relation!)
* Try out different loss function
* Evaluate the test data and training data and see the accuracy 
* Experiment with the code

# Report Guidelines
In the field of Computer Science, most of the latest techniques are proposed at conferences instead of journal, like NIPS, ACL, CVPR, AAAI and so on.

We will follow IEEE guidelines to write paper report.
A proper report will contain-
 * Abstract
 * Introduction
 * Methods
 * Experiment
 * Results
 * Conclusion 
 * References
 For more details: https://www.ieee.org/conferences/publishing/templates.html
 We recommend to use overleaf latex Editor.

# Grading
 Grading is based on your report.
Below are the grading policy for this homework:
* The report should be in English.
* Introduction & Conclusion (30%)
* Methods (30%)
* Experiment & Results (30%)
* Abstract & References (10%)

# Final Project
We suggest selecting one interesting topic as early as possible. Most of the students here are undergraduate or new to coding, it would be the best time for you to develop your knowledge and the ability to implement code by reading research papers.

# Papers
You can find papers here:
* https://arxiv.org
* Google Scholar
* Conference Websites
"""

