# Accuracy chart

Dataset: CIFAR10

## Size of dataset

Type|Size|
|---|---|
|Training dataset| 50000|
|Test dataset| 10000|
|Total | 60000|

Image size: 32x32

|Type|Epoch|Batch size | Loss (epoch)|Test Accuracy | Machine used |
|---|---|---|---|---|---|
|Original code | 2 | 16 | 1.140639 | 60.26% | Linux |
|ResNet (n=2) | 2 | 16 | 1.239102 | 57.14% | Linux |
|ResNet (n=2) | 4 | 16 | 0.600146 | 59.07% | Linux |
|ResNet (n=2) | 8 | 16 | 0.175138 | 57.46% | Linux |
|ResNet (n=2) | 16 | 16 | 0.110585 | 57.73% | Linux |
|ResNet (n=2) | 16 | 128 | 0.050868 | 60.77% | Linux |
|ResNet (n=2) | 32 | 128 | 0.031375 | 60.95% | Linux |
|ResNet (n=2) with InstanceNorm | 8 | 128 | 0.488496 | 67.16% | Linux |
|ResNet (n=2) with InstanceNorm | 32 | 128 | 0.050318 | 69.46% | Linux |
|ResNet (n=5) with InstanceNorm | 32 | 128 | 0.063170 | 70.70% | Linux |
|ResNet (n=5) with InstanceNorm | 100 | 128 | 0.017879 | 71.40% | Linux |
|ResNet (n=9) with InstanceNorm | 100 | 128 | 0.036436 | 70.41% | Linux |
|ResNet (n=9) with InstanceNorm | 100+100 | 128 | 0.018815 | 71.86% | Linux |
|ResNet (n=2) with BatchNorm | 8 | 128 | 0.188598 | 59.10% | Linux |
|ResNet (n=2) with BatchNorm | 32 | 128 | 0.027148 | 61.59% | Linux |
|ResNet (n=2) with BatchNorm and global average pool | 8 | 128 | 0.801303 | 67.65% | Linux |
|ResNet (n=2) with BatchNorm and global average pool | 32 | 128 | 0.188459 | 70.27% | Linux |

## Machines used
* Mac (OS:10.13.5, RAM: 16 GB, CPU: 2.6 GHz Intel Core i5, Python: 3.6.7) 