# sd-webui-diffpure

## Diffusion Models for Adversarial Purification

**Auto1111** port of NVlab's adversarial purification method that uses the forward and reverse processes of diffusion models to remove adversarial perturbations.

**WIP**: the script is installable and launchable, but not yet tested by me with a pretrained model. Stay tuned.

## How to use?

Go to img2img tab, select the DiffPure script. Download a model and a config from the links referenced there. Drop your picture with adversarial noise into the image input window and then just run the img2img process.

## Info

See the source repository https://github.com/NVlabs/DiffPure and the project page https://diffpure.github.io/ by Weili Nie, Brandon Guo, Yujia Huang, Chaowei Xiao, Arash Vahdat, Anima Anandkumar for more info.

![scheme](https://github.com/NVlabs/DiffPure/raw/master/assets/teaser_v7.jpeg)

> Abstract: Adversarial purification refers to a class of defense methods that remove adversarial perturbations using a generative model. These methods do not make assumptions on the form of attack and the classification model, and thus can defend pre-existing classifiers against unseen threats. However, their performance currently falls behind adversarial training methods. In this work, we propose DiffPure that uses diffusion models for adversarial purification: Given an adversarial example, we first diffuse it with a small amount of noise following a forward diffusion process, and then recover the clean image through a reverse generative process. To evaluate our method against strong adaptive attacks in an efficient and scalable way, we propose to use the adjoint method to compute full gradients of the reverse generative process. Extensive experiments on three image datasets including CIFAR-10, ImageNet and CelebA-HQ with three classifier architectures including ResNet, WideResNet and ViT demonstrate that our method achieves the state-of-the-art results, outperforming current adversarial training and adversarial purification methods, often by a large margin.

![images](https://diffpure.github.io/assets/smile_glasses_v0.jpeg)

## License

Please check the LICENSE file. This work may be used non-commercially, meaning for research or evaluation purposes only. For business inquiries, please contact researchinquiries@nvidia.com.

### TODO: make a better readme