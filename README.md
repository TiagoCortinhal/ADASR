# SR_GAN
Internship Project: Unsupervised Super Resolution With GAN 

This projects aims to tackle some issues with Super Resolution namely the possible lack of training data. 
We use Domain Adaptation techniques to tackle this issue and the lack of high resolution data by not only adapt to a different
domain (let's say landscapes and faces) but also the adapt for the resolution, making the extracted features closer both to the domains
and the different resolutions available.


We started by adapting EDSR network to use Wasserstein GAN (with gradient penalty) which produces better results and it's less prone to modal collapses.

We used the ideas behind DANN to introduce this double domain adaptation.

The datasets used are DIV2K (from NTIRE) and Celeba-HQ. We created patch files based on the resolution (which are heavy to be uploaded here, script will be added soon) to 
make the trainning easier. 

Other datasets can be used and tested, and it is easy to change the dataloader to create patches directly.

Results coming soon.


# References 

 * [EDSR](https://arxiv.org/pdf/1707.02921.pdf) - [code](https://github.com/thstkdgus35/EDSR-PyTorch)
 * [DANN](https://arxiv.org/pdf/1505.07818.pdf) - [code](https://github.com/CuthbertCai/pytorch_DANN)
 * [WGAN](https://arxiv.org/pdf/1701.07875.pdf) - [one of many repos consulted](https://github.com/caogang/wgan-gp/blob/master/gan_toy.py)
 * [ADDA](https://arxiv.org/pdf/1702.05464.pdf) - [one of many implementations](https://github.com/corenel/pytorch-adda)
 * [NTIRE](http://www.vision.ee.ethz.ch/ntire19/)
 * [Celeba-HQ](https://github.com/nperraud/download-celebA-HQ)
