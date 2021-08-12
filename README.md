# Latext-based-EnhancerGAN
#### Note: This study is inspired by the work of LATEXT-GAN

This work is the extension of the original Enahcner GAN. LA-based-enhancer-GAN tries to solve the problems faced by 1d Conv based enhancer GAN.


### Problems:
- Not every enhancer produced is present in the human genome
- more training time
- Unstable training


### LA-based-enhancer-GAN:
- A pretrained AE to learn the continuous representation of AEs]
- Less training data 
- Less training time
- More accurate learning of Enhancer regions



### Dataset:
- 43011 experimentally defined enhancers from human genome


### How to use it ?
- Run the train_ae.py
- Get the pretrained AE model
- Run the test_ae.py
- Run the train_gan.py
- Get the results
- Do blast
- Perform biological analyses

### Results
- The work successfuly generates enhancers of similar size e.g. in our case it prodcues 131 Nucs Enhancers etc.
  ##### Loss:

<p align="middle"> 
  <img src="/closs.png" width="400" />
  <img src="/gloss.png" width="400" /> 
  <img src="/ae_loss.png" width="400" />
</p>

##### Alignment of the generated Enhancers:

<p align="middle"> 
  <img src="/al1.svg" width="800" />
</p>
