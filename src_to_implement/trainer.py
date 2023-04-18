import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import os

class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 train_crit,                         # Loss function
                 val_crit=None,
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 scheduler=None,               # Scheduler
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._train_crit = train_crit
        self._val_crit=val_crit
        self._optim = optim
        self._schedlr=scheduler
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda and t.cuda.is_available():
            self._model = model.cuda()
            self._train_crit = train_crit.cuda() if self._train_crit is not None else None
            self._val_crit = val_crit.cuda() if self._val_crit is not None else None
            
    def save_checkpoint(self, epoch):
        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if (self._cuda and t.cuda.is_available()) else 'cpu')
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
        
    def modify_Linear(self, lr=None, wd=None, out_features=0):
        if out_features>0:
            #self._model.fc = t.nn.Sequential(t.nn.Dropout(p=0.5), t.nn.Linear(512,out_features), t.nn.ReLU(inplace=True), t.nn.Dropout(p=0.5), t.nn.Linear(out_features,out_features), t.nn.ReLU(inplace=True), t.nn.Linear(out_features, 2))
            #self._model.add_module('relu_out',t.nn.ReLU(inplace=True))
            '''for param in self._model.layer4.parameters():
                print(param)'''
            for param in self._model.parameters():
                param.requires_grad = False
            '''for param in self._model.fc.parameters():
                param.requires_grad = True'''
            #self._optim=t.optim.Adam(self._model.fc.parameters(),lr=lr,weight_decay=wd)
            self._optim=t.optim.SGD(self._model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
            '''for param in self._model.layer4.parameters():
                print(param)'''
            '''for name,layer in self._model.named_children():
                print(name,layer)'''
        else:
            self._model.add_module('sigmoid',t.nn.Sigmoid())
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        self._optim.zero_grad()
        # -propagate through the network
        pred = self._model(x)
        #pred = t.sigmoid(self._model(x))
        # -calculate the loss
        loss = self._train_crit(pred, y.float())
        # -compute gradient by backward propagation
        loss.backward()
        # -update weights
        self._optim.step()
        # -return the loss
        return loss.item()
        #TODO
        
        
    
    def val_test_step(self, x, y):
        
        # predict
        pred = self._model(x)
        #pred = t.sigmoid(self._model(x))
        # propagate through the network and calculate the loss and predictions
        loss = self._val_crit(pred, y.float()) if self._val_crit is not None else self._train_crit(pred, y.float())
        # return the loss and the predictions
        return loss.item(), pred
        #TODO
        
    def train_epoch(self):
        # set training mode
        self._model = self._model.train()
        # iterate through the training set
        loss = 0
        for img, label in self._train_dl:
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda and t.cuda.is_available():
                img = img.to('cuda')
                label = label.to('cuda')
            else:
                img = img.to('cpu')
                label = label.to('cpu')
        # perform a training step
            loss = loss + self.train_step(x=img, y=label)        
        # calculate the average loss for the epoch and return it
        avg_loss = loss / len(self._train_dl)
        return avg_loss
        #TODO
    
    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        self._model = self._model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore.
        with t.no_grad():
            total_loss = 0
            preds = None
            labels = None
            # iterate through the validation set
            for img, label in self._val_test_dl:
            # transfer the batch to the gpu if given
                if self._cuda and t.cuda.is_available():
                    img = img.to('cuda')
                    label = label.to('cuda')
                else:
                    img = img.to('cpu')
                    label = label.to('cpu')
            # perform a validation step
                loss, pred = self.val_test_step(img, label)
                pred=t.sigmoid(pred).round()   #when BCEwithLogitsLoss is used
                #pred = pred.round()
                total_loss = total_loss + loss
            # save the predictions and the labels for each batch
                if preds is None and labels is None:
                    labels = label
                    preds = pred
                else:
                    labels = t.cat((labels, label), dim=0)
                    preds = t.cat((preds, pred), dim=0)
            # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
            avg_loss=total_loss / len(self._val_test_dl)
            F1_crack = f1_score(labels[:, 0].cpu(), preds[:, 0].cpu(), average='binary')
            F1_inactive = f1_score(labels[:, 1].cpu(), preds[:, 1].cpu(), average='binary')
            F1_mean = (F1_crack + F1_inactive) / 2
            self.f1_score=F1_mean

            # return the loss and print the calculated metrics
            if self._schedlr is not None:
                #self._schedlr.step()
                #self._schedlr.step(avg_loss) #to be used when ReduceLRonPlateau is scheduler
                for s in self._schedlr[:-1]:   #to be used when multiple scheduler
                    s.step()
                self._schedlr[-1].step(avg_loss)
            print("F1_Crack={},F1_inactive={},F1_mean={}".format(F1_crack,F1_inactive,F1_mean))
            return avg_loss
        #TODO

    def visualize_output(self,x,y):
        import matplotlib.pyplot as plt
        import numpy as np
        self._model = self._model.eval()
        # Attach hooks to conv layers
        activations = {}
        def get_activation(name):
            def conv_hook(model, input, output):
                # Remove batch dimension (=1) and calculate mean of feature maps along the channel dimension
                activations[name] = output.detach().squeeze(0).mean(0).numpy()
            return conv_hook

        for module in self._model.modules():
            if isinstance(module, t.nn.Conv2d):
                module.register_forward_hook(get_activation(f"Conv {module.in_channels}, {module.out_channels}"))
        x = np.expand_dims(x, axis=0)
        x=t.tensor(x)
        pred = self._model(x)
        loss = self._train_crit(pred[0], y.float())
        loss.backward()
        # Plot activations
        fig = plt.figure(figsize=(30, 50))
        for i, (name, feature_map) in enumerate(activations.items()):
            a = fig.add_subplot(7, 7, i+1)
            plt.imshow(feature_map)
            a.axis("off")
            a.set_title(name, fontsize=30)
        plt.savefig('feature_maps.png', bbox_inches='tight')
        print("Label=",y)
        pred = t.sigmoid(self._model(x))[0]
        print("Pred=",pred)

        
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        train_losses = []
        val_losses = []
        epoch_cntr = 0
        patience_cntr=0
        val_loss_min=None
        f1_max=0
        self.f1_scores=[]
        #TODO
        
        while True:
      
            # stop by epoch number
            if epoch_cntr == epochs:
                break
            # train for a epoch and then calculate the loss and metrics on the validation set
            epoch_cntr += 1
            train_loss = self.train_epoch()
            val_loss = self.val_test()
            # append the losses to the respective lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            self.f1_scores.append(self.f1_score)
            if val_loss_min is None:
                val_loss_min=val_loss
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            if (len(val_losses)>0 and val_loss <= val_loss_min) or (self.f1_score>=f1_max):
                self.save_checkpoint(epoch_cntr)
                #self.save_onnx('checkpoint_{:03d}.onnx'.format(epoch_cntr))
                patience_cntr=0
                f1_max=self.f1_score
                val_loss_min=val_loss
            elif ((len(val_losses) >1 and val_loss > 1.02 * val_losses[-2]) or (self.f1_score<f1_max)):
                patience_cntr += 1
            print("Train_loss={},Val_loss={}".format(train_loss,val_loss))
            print("Epoch counter={},Patience counter={},f1_max={}\n".format(epoch_cntr,patience_cntr,f1_max))
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            if epoch_cntr==epochs or (self._early_stopping_patience>0 and patience_cntr==self._early_stopping_patience):
                return train_losses,val_losses
            # return the losses for both training and validation
        #TODO