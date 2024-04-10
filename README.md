scTPC: a novel semi-supervised deep clustering model for scRNA-seq data
=
This study investigates a semi-supervised clustering model called scTPC, which integrates the triplet constraint, pairwise constraint and cross-entropy constraint based on deep learning. Specifically, the model begins by pre-training a denoising autoencoder based on a zero-inflated negative binomial (ZINB) distribution. Deep clustering is then performed in the learned latent feature space using triplet constraints and pairwise constraints generated from partial labeled cells. Finally, to address imbalanced cell-type datasets, a weighted cross-entropy loss is introduced to optimize the model. A series of experimental results on 10 real scRNA-seq datasets and 5 simulated datasets demonstrate that scTPC achieves accurate clustering with a well-designed framework.

Framework diagram  
=
![](https://github.com/LF-Yang/Code/blob/master/Framework.png)

Requirements  
=
Python---3.8  
PyTorch---1.12  
scanpy---1.9.2  
It should be noted that the results obtained may vary slightly depending on the version used.  

Usage  
=
main.py---Execute the main program file to obtain clustering performance results and representations of latent features.  
scNetwork.py---The framework of the model.  
Generate.py---Create triplets of elements.  
Preprocess.py---Perform preprocessing operations on the input data.  
Layers.py---Specify the layers of the network.  

To obtain the results, simply run the "main.py" file as the dataset is in h5 format. The final output reports the clustering performance ( NMI, ARI, ACC and AMI ). By replacing "filename", you can obtain clustering metrics for different datasets.  
