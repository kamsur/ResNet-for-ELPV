import sys

from torchvision import models
import torch as t
import pandas as pd

import model
from trainer import Trainer
from data import ChallengeDataset

# Adapt this to match the model used in train.py
model = models.resnet34()
model.fc = t.nn.Sequential(t.nn.Linear(512, 2))

# Load weights of given epoch
epoch,img = int(sys.argv[1]),int(sys.argv[2])
crit = t.nn.BCEWithLogitsLoss()
trainer = Trainer(model, crit, None)
trainer.restore_checkpoint(epoch)

# Load dataset
dataframe = pd.read_csv('data.csv', sep=';')
dataset = ChallengeDataset(dataframe, "val")


# Pick sample for which you want to visualize the activations
x, y = dataset[img]
trainer.visualize_output(x,y)