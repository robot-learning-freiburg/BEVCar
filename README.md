# BEVCar
[**arXiv**](https://arxiv.org) | [**Website**](http://bevcar.cs.uni-freiburg.de/) | [**Video**](https://youtu.be/bB_k_6IvPHQ?feature=shared)

This repository is the official implementation of the paper:

> **BEVCar: Camera-Radar Fusion for BEV Map and Object Segmentation**
>
> [Jonas Schramm]()&ast;, [Niclas Vödisch](https://vniclas.github.io/)&ast;, [Kürsat Petek](http://www2.informatik.uni-freiburg.de/~petek/)&ast;, [B Ravi Kiran](), [Senthil Yogamani](), [Wolfram Burgard](https://www.utn.de/person/wolfram-burgard/), and [Abhinav Valada](https://rl.uni-freiburg.de/people/valada). <br>
> &ast;Equal contribution. <br> 
> 
> *under review*, 2024

<p align="center">
  <img src="./assets/bevcar_overview.png" alt="Overview of BEVCar approach" width="800" />
</p>

If you find our work useful, please consider citing our paper:
```
@article{schramm2024bevcar,
  title={BEVCar: Camera-Radar Fusion for BEV Map and Object Segmentation},
  author={Schramm, Jonas and Vödisch, Niclas and Petek, Kürsat and Kiran, B Ravi and Yogamani, Senthil and Burgard, Wolfram and Valada, Abhinav},
  journal={},
  year={2024}
}
```


## 📔 Abstract

Semantic scene segmentation from a bird's-eye-view (BEV) perspective plays a crucial role in facilitating planning and decision-making for mobile robots. Although recent vision-only methods have demonstrated notable advancements in performance, they often struggle under adverse illumination conditions such as rain or nighttime. While active sensors offer a solution to this challenge, the prohibitively high cost of LiDARs remains a limiting factor. Fusing camera data with automotive radars poses a more inexpensive alternative but has received less attention in prior research. In this work, we aim to advance this promising avenue by introducing BEVCar, a novel approach for joint BEV object and map segmentation. The core novelty of our approach lies in first learning a point-based encoding of raw radar data, which is then leveraged to efficiently initialize the lifting of image features into the BEV space. We perform extensive experiments on the nuScenes dataset and demonstrate that BEVCar outperforms the current state of the art. Moreover, we show that incorporating radar information significantly enhances robustness in challenging environmental conditions and improves segmentation performance for distant objects.


## 👩‍💻 Code

We will release the code upon the acceptance of our paper.


## 👩‍⚖️  License

The code is released under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.
For any commercial purpose, please contact the authors.


## 🙏 Acknowledgment

This work was funded by Qualcomm Technologies Inc. and the German Research Foundation (DFG) Emmy Noether Program grant No 468878300.
<br><br>
<p float="left">
  <a href="https://www.qualcomm.com/"><img src="./assets/qualcomm_logo.png" alt="drawing" height="80"/></a>
  &nbsp;
  &nbsp;
  &nbsp;
  <a href="https://www.dfg.de/en/research_funding/programmes/individual/emmy_noether/index.html"><img src="./assets/dfg_logo.png" alt="DFG logo" height="100"/></a>
</p>
