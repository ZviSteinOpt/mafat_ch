import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,WeightedRandomSampler,random_split

data = pd.read_csv('/Users/zvistein/Downloads/mafat_wifi_challenge_training_set_v1.csv')
l = (len(data.RSSI_Right)-len(data.RSSI_Right)%360)
b =  data.RSSI_Right[0:l]
a =  data.RSSI_Left[0:l]
a = np.append(a, b)
rss =  a.reshape(int(l/360*2),360)
rss_n = np.zeros((int(l/360),360,2))
rss_n[:,:,0] = rss[0:int(l/360),:]
rss_n[:,:,1] = rss[int(l/360):,:]

b = data.Num_People[0:l]
num = b.values.reshape(int(l/360),360)
gt = np.zeros(int(l/360))
for i in np.arange(0,int(l/360)):
 n = num[i,:]
 b = Counter(n)
 gt[i] = b.most_common(1)[0][0]

train_data = []
for i in range(len(rss_n)):
   train_data.append([rss_n[i], gt[i]])


class FN(nn.Module):
    def __init__(self, num_classes = 4):
        super(FN, self).__init__()

        # Output size after convolution filter
        # ((w-f+2P)/s) +1

        # Input shape= (b_s,1,50,50)

        self.fc1 = nn.Linear(in_features=360*2, out_features=36000)
        self.fc2 = nn.Linear(in_features=36000, out_features = 10000)
        self.fc3 = nn.Linear(in_features=10000, out_features = 1000)
        self.fc4 = nn.Linear(in_features=1000, out_features = 100)
        self.fc5 = nn.Linear(in_features=100, out_features = 4)


        self.relu = nn.ReLU()

        # Feed forwad function

    def forward(self, input):
        output = self.fc1(input)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        output = self.relu(output)
        output = self.fc4(output)
        output = self.relu(output)
        output = self.fc5(output)
        return output


test_count  = len(rss_n)//7
train_count = len(rss_n)-test_count

train_sets, test_setes = random_split(train_data,[train_count,test_count])

test_loader = DataLoader(test_setes,
    batch_size=1, shuffle=True)
train_loader = DataLoader(train_sets,
    batch_size=50, shuffle=True)


model = FN(num_classes=4)

optimizer     = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=(0.9))

loss_function = nn.CrossEntropyLoss()

num_epochs = 10
s_loss = torch.zeros(num_epochs*len(train_loader))
idx = 0

for epoch in range(num_epochs):

    # Evaluation and training on training dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0

    for i, (seq, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        outputs = model(seq.view(-1,360*2).float())
        loss = loss_function(outputs, labels.long())
        s_loss[idx] = loss
        idx = idx+1
        loss.backward()
        optimizer.step()
        print(loss)