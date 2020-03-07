# DCGAN
Deep Convolutional Generative Adversarial Network[1]

![dcgan](_img/dcgan.png)

Generating cat images with DCGANS on hops.site because who doens't love cats?
Showing results from running a while on a gpu before computaional budget expired.

The architecture had to be adjusted a little bit to get any learning signal...

## Results

![cat_images](_results/_generated_cat_images/Screen Shot 2018-01-08 at 19.21.07.png)


## Training Curves

![d_hyperparam_search](_results/_figures/d_hyperparam_search.png)

![g_hyperparam_search](_results/_figures/g_hyperparam_search.png)

![d_loss](_results/_figures/d_loss.png)

![g_loss](_results/_figures/g_loss.png)

## Requirements 

* [Tensorflow 0.12.1](https://github.com/tensorflow/tensorflow/tree/r0.12)

## Usage

> python train.py

or simply run the jupyter notebook...

## Disclaimer

Old project from 2017 or something... 
code quaility is shit

## References

[1] https://arxiv.org/pdf/1511.06434.pdf
