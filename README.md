# Solar cell defect detection
# A multi class image classification problem on ELPV dataset
This is a PyTorch implementation of a modified ResNet-34 architecture, for a competition (involving all participants of Deep Learning course at FAU Erlangen Nuremberg, Winter Semester 22/23) to detect multi-label defects in Electroluminescence images of photovoltaic modules (with only 2000 samples of ELPV dataset used for training and testing).
The images are labelled using two-bit binary code, denoting presence of two mutually non-exclusive defects, 'inactive' and 'crack'. Class imbalance was present and weighted loss is used to compensate for this during training.
ResNet-34, pretrained on ImageNet dataset, with two outputs in the Fully Connected layer, is used for the classification task with Sigmoid activation function.
BinaryCrossEntropy Loss is used as loss function and ADAM optimizer is used without weight decay(L2 regularization).
