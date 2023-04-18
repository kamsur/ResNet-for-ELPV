import torch as t
from trainer import Trainer
import sys
import torchvision as tv
#import model

epoch = int(sys.argv[1])
#TODO: Enter your model here
#model=model.ResNet()
model=tv.models.resnet34()
model.fc = t.nn.Sequential(t.nn.Linear(512, 2))
crit = t.nn.BCEWithLogitsLoss()
trainer = Trainer(model, crit, None)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))
