# High-Fidelity Pluralistic Image Completion with Transformers

<img src='imgs/teaser.png'/>

### [Project Page](http://raywzy.com/ICT/) | [Paper (ArXiv)](https://arxiv.org/abs/2004.09484)


[Ziyu Wan](http://raywzy.com/)<sup>1</sup>,
[Jingbo Zhang]()<sup>1</sup>,
[Dongdong Chen](http://www.dongdongchen.bid/)<sup>2</sup>,
[Jing Liao](https://liaojing.github.io/html/)<sup>1</sup> <br>
<sup>1</sup>City University of Hong Kong, <sup>2</sup>Microsoft Cloud AI


## Abstract
<img src='imgs/pipeline.pdf'/>
Image completion has made tremendous progress with convolutional neural networks (CNNs), because of their powerful texture modeling capacity. However, due to some inherent properties (\eg, local inductive prior, spatial-invariant kernels), CNNs do not perform well in understanding global structures or naturally support pluralistic completion. Recently, transformers demonstrate their power in modeling the long-term relationship and generating diverse results, but their computation complexity is quadratic to input length, thus hampering the application in processing high-resolution images. This paper brings the best of both worlds to pluralistic image completion: appearance prior reconstruction with transformer and texture replenishment with CNN. The former transformer recovers pluralistic coherent structures together with some coarse textures, while the latter CNN enhances the local texture details of coarse priors guided by the high-resolution masked images. The proposed method vastly outperforms state-of-the-art methods in terms of three aspects: 1) large performance boost on image fidelity even compared to deterministic completion methods; 2) better diversity and higher fidelity for pluralistic completion; 3) exceptional generalization ability on large masks and generic dataset, like ImageNet.



## To Do
- [ ] Release testing code
- [ ] Release pretrained model
- [ ] Release training code


## Citation

If you find our work useful for your research, please consider citing the following papers :)

```

```
