# SeFa - Closed-Form Factorization of Latent Semantics in GANs

![image](./docs/assets/teaser.jpg)
**Figure:** *Versatile semantics found from various types of GAN models using SeFa.*

> **Closed-Form Factorization of Latent Semantics in GANs** <br>
> Yujun Shen, Bolei Zhou <br>
> *arXiv preprint arXiv:2007.06600*

[[Paper](https://arxiv.org/pdf/2007.06600.pdf)]
[[Project Page](https://genforce.github.io/sefa/)]
[[Demo](https://www.youtube.com/watch?v=OFHW2WbXXIQ)]

In this repository, we propose a *closed-form* approach, termed as **SeFa**, for *unsupervised* latent semantic factorization in GANs. With this algorithm, we are able to discover versatile semantics from different GAN models trained on various datasets. Most importantly, the proposed method does *not* rely on pre-trained semantic predictors and has an extremely *fast* implementation (*i.e.*, less than 1 second to interpret a model). Below show some interesting results on anime faces, cats, and cars.

**NOTE:** The following semantics are identified in a completely *unsupervised* manner, and post-annotated for reference.

| Anime Faces | | |
| :-- | :-- | :-- |
| Pose | Mouth | Painting Style
| ![image](./docs/assets/stylegan_animeface_pose.gif) | ![image](./docs/assets/stylegan_animeface_mouth.gif) | ![image](./docs/assets/stylegan_animeface_style.gif)

| Cats | | |
| :-- | :-- | :-- |
| Posture (Left & Right) | Posture (Up & Down) | Zoom
| ![image](./docs/assets/stylegan_cat_posture_horizontal.gif) | ![image](./docs/assets/stylegan_cat_posture_vertical.gif) | ![image](./docs/assets/stylegan_cat_zoom.gif)

| Cars | | |
| :-- | :-- | :-- |
| Orientation | Vertical Position | Shape
| ![image](./docs/assets/stylegan_car_orientation.gif) | ![image](./docs/assets/stylegan_car_vertical_position.gif) | ![image](./docs/assets/stylegan_car_shape.gif)

## BibTeX

```bibtex
@article{shen2020closedform,
  title   = {Closed-Form Factorization of Latent Semantics in GANs},
  author  = {Shen, Yujun and Zhou, Bolei},
  journal = {arXiv preprint arXiv:2007.06600},
  year    = {2020}
}
```

## Code Coming Soon
