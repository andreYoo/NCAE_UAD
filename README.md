# NCAE_UAD
Official project repository of 'Normality-Calibrated Autoencoder for Unsupervised Anomaly Detection on Data Contamination'

## Abstract
In this paper, we propose Normality-Calibrated Autoencoder (NCAE), which can boost anomaly detection performance on the contaminated datasets without any prior information or explicit abnormal samples in the training phase. The NCAE adversarially generates high confident normal samples from a latent space having low entropy and leverages them to predict abnormal samples in a training dataset. NCAE is trained to minimise reconstruction errors in uncontaminated samples and maximise reconstruction errors in contaminated samples. The experimental results demonstrate that our method outperforms shallow, hybrid, and deep methods for unsupervised anomaly detection and achieves comparable performance compared with semi-supervised methods using labelled anomaly samples in the training phase. The source code is publicly available on this website


## Dependencies
This project mainly complied with Python3.6, Pytorch 1.3. All details are included in the 'requirement.txt'


## Training and Testing
See the './src/run.sh' file



## Code reference
* The code is mainly encouraged by [DeepSAD](https://github.com/lukasruff/Deep-SAD-PyTorch) 


