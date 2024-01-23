# LeeJeongHwi  Master's Thesis

### Title : Interpretable Arrhythmia Type Classification Using Vision Transformer Model



### Environment

* Python 3.10.13

* Pytorch 2.1.0
* Scikit-learn 1.3.2
* numpy 1.26.0
* pandas 2.1.3
* ecg-plot 0.2.8
* einops 0.7.0
* pytorch-model-summary 0.1.2
* scipy 1.11.4
* seaborn 0.13.0



### Database

[PTB-XL, a large publicly available electrocardiography dataset](https://physionet.org/content/ptb-xl/1.0.3/) 1.0.3



### Model Summary

![model](Figure/model.png)



### Filtering

**Highpass Filter (for Powerline Noise Remove)**

Before

<img src="Figure/freq_before.png" alt="freq_before" style="zoom:50%;" />

After

<img src="Figure/freq_af.png" alt="freq_af" style="zoom:50%;" />



**Median Filter (for Baseline Wander Noise Remove)**

Before

![Data_wavelet_Bf](Figure/Data_wavelet_Bf.png)

After

![Data_wavelet_af](Figure/Data_wavelet_af.png)



### Train

Train Loss

<img src="Figure/Train_loss.png" alt="Train_loss" style="zoom:75%;" />



### Evaluation

<img src="Figure/cfmt.png" alt="Train_loss" style="zoom:75%;" />



### Visualization

Conduction Disturbance (Type : LBBB)

<img src="Figure/head12.png" alt="head12" style="zoom: 50%;" />





---

Reference

https://github.com/yonigottesman/ecg_vit

