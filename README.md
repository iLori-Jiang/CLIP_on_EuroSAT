# L'X INF649 computer vision course project

## Authors
- **Haiyang Jiang** - (haiyang.jiang@polytechnique.edu)
- **Jingnan Cao** - (jingnan.cao@polytechnique.edu)

## Abstract
Few shots (16 shots) learning CLIP on EuroSAT dataset, by linear probing and prompt engineering.

## Explanation

Each Jupyter Notebook records the process of our implementation. For linear probe, the file contains how we train the model and test for the result. For prompt engineering using CoOp and CoCoOp, the model is trained by the official repository: (https://github.com/KaiyangZhou/CoOp), and then the model of CoOp tested by the notebook. The CoCoOp model is tested by the official test pipeline during training, where the log of the testing can be seen in the "models/eurosat-CoCoOp/test_new/../log.txt". The training can also be refered to the training "log.txt" under each model directory.

## Detail

For both methods we apply 16 shots learning.

For linear probe, we simply add a linear layer from the output of the image encoder (ViT-B/32) to the number of class, as a single layer classifier head.

For prompt engineering, for CoOp based on ResNet50, we use 16 context length, with class word at the end of the context, which compose a 17-word query sentence in total. We train for both class-specified-context (CSC: True), which is the context is specified for different classes, and no class-specified-context (CSC: False) or so-called unified context, which is the same context for all the classes. For CoCoOp based on ViT-B/16, the context length is 4, with class word at the end of the context, which compose a 5-word query sentence in total. And there is a nueral network to transform each image into image token to be added to the context.

