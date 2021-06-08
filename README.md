# MTL-Sign
This repository contains materials for the paper entitled "MTLSign: Secure Watermarking Framework for Deep Neural Networks with Multi-task Learning"

Data should be downloaded and unzipped at ./data/XXX.
We released the QR codes adopted as the watermarking dataset.
Other datasets can be downloaded from torchvision or their corresponding sources.
For semantic segmentation tasks, you need the COCO tools, which have been provided.
For NLP tasks, you need the GLOVE pre-trained embedding, which is available at https://nlp.stanford.edu/projects/glove/.

Environment:
PyTorch>=1.7.1
CUDA>=10.1

For MTLSign on image classification models, see Feb_mnist.py.
-The member function mnist_acc() calculates the model's performance on the primary task.
-The member function verify_acc() calculates the model's performance on the watermarking task.

For MTLSign on image segmentation models, see Feb_fdp.py.
-The function verify_acc() calculates the model's performance on the watermarking task.

For MTLSign on NLP models, see Feb_sa.py.
-The member function wm_acc() calculates the model's performance on the watermarking task.

