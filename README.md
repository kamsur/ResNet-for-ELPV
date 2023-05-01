Solar cell defect detection
===============================================================================================================================
A multi class image classification problem on ELPV dataset
-------------------------------------------------------------------------------------------------------------------------------
1. This is a **PyTorch** implementation of a modified **ResNet-34** architecture, for a competition (involving all participants of Deep Learning course at FAU Erlangen Nuremberg, Winter Semester 22/23) to detect multi-label defects in Electroluminescence images of photovoltaic modules (with only *2000* samples of **ELPV** dataset used for training and testing).
2. The images are labelled using two-bit binary code, denoting presence of two mutually non-exclusive defects, **'inactive'** and **'crack'**. <ins>Class imbalance</ins> was present and **weighted loss** is used to compensate for this during training.
3. ResNet-34, pretrained on ImageNet dataset, with two outputs in the Fully Connected layer, is used for the classification task with **Sigmoid** activation function.
4. **BinaryCrossEntropy Loss** is used as loss function and **ADAM** optimizer is used without weight decay(L2 regularization).
5. The model stood at **top 3** position among 120+ participants (mean f1 score performance for the two defect classes, on a separate validation set was used for judging winners)
