# Latext-based-EnhancerGAN

This work is the extension of the original Enahcner GAN. LA-based-enhancer-GAN tries to solve the problems faced by 1d Conv based enhancer GAN.


Problems:
- Not every enhancer produced is present in the human genome
- more training time
- Ustable training


LA-based-enhancer-GAN:
- A pretrained AE to learn the continuous representation of AEs]
- Less training data 
- Less training time
- More accurate learning of Enhancer regions



Dataset:
- 43011 experimentally defined enhancers from human genome


How to use it ?
- Run the train_ae.py
- Get the pretrained AE model
- Run the test_ae.py
- Run the train_gan.py
- Get the results
- Do blast
- Perform biological analyses
