import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
#import model
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from torchvision import models

'''t.manual_seed(0)
t.backends.cudnn.benchmark = False
t.backends.cudnn.deterministic = True'''
# load the data from the csv file and perform a train-test-split
csv_path = ''
for root, _, files in os.walk('.'):
    for name in files:
        if name == 'data.csv':
            csv_path = os.path.join(root, name)
dataFrame = pd.read_csv(csv_path, sep=';')
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
'''class_map = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
stratify_labels = [class_map[(x, y)] for x, y in dataFrame[['crack', 'inactive']].to_numpy()]'''
#train_dF, val_dF = train_test_split(dataFrame, test_size=0.1, shuffle=True, random_state=42, stratify=stratify_labels)
train_dF, val_dF = train_test_split(dataFrame, test_size=0.2, random_state=42)
#train_dF.reset_index(inplace=True)
#val_dF.reset_index(inplace=True)
train_dataset=ChallengeDataset(train_dF, 'train')
val_dataset=ChallengeDataset(val_dF, 'val')
train_batch_size=32
val_batch_size=32
# TODO

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_dl = t.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, drop_last=False, shuffle=True)
val_dl = t.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, drop_last=True, shuffle=False)
#train_dl = t.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
#val_dl = t.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)
# TODO

# create an instance of our ResNet model
#resNet=model.ResNet()
resNet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
resNet.fc = t.nn.Sequential(t.nn.Linear(512, 2))
'''for name,layer in resNet.named_children():
    print(name,layer)'''
'''for param in resNet.parameters():
    param.requires_grad = True'''
for param in resNet.fc.parameters():
    param.requires_grad = True
# TODO

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
train_class_weight=train_dataset.calc_class_weight()
#val_class_weight=val_dataset.calc_class_weight()
train_lossCrit=t.nn.BCEWithLogitsLoss(pos_weight=train_class_weight)
#val_lossCrit=t.nn.BCEWithLogitsLoss(pos_weight=val_class_weight)
#crit = t.nn.BCELoss()
# set up the optimizer (see t.optim)
learning_rate,weight_decay=(7e-5,0)
optimizer=t.optim.Adam(resNet.parameters(),lr=learning_rate,weight_decay=weight_decay)
#optimizer = t.optim.SGD(resNet.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)


#scheduler1 = t.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20 , 30 , 40], gamma=0.1)
#scheduler2 = t.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80, 100, 130], gamma=0.5)
#scheduler1 = t.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7,13,19], gamma=0.4, verbose=True)
#scheduler2 = t.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12,15,20], gamma=0.8, verbose=True)
scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=5,verbose=True)
#scheduler = t.optim.lr_scheduler.ExponentialLR(optimizer,0.9,verbose=True)
# create an object of type Trainer and set its early stopping criterion
trainer=Trainer(resNet,train_lossCrit,None,optimizer,train_dl,val_dl,scheduler=[scheduler],cuda=True,early_stopping_patience=7)
# TODO
#trainer.restore_checkpoint(13)
# go, go, go... call fit on trainer
res = trainer.fit(epochs=50)#TODO

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
#plt.savefig('losses.png')

plt.savefig('losses_lr={}_wd={}.png'.format(str(learning_rate),str(weight_decay)))
plt.figure()
plt.plot(np.arange(len(trainer.f1_scores)), trainer.f1_scores, label='f1 score')
plt.savefig('f1_lr={}_wd={}.png'.format(str(learning_rate),str(weight_decay)))