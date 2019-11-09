from imgaug import augmenters as iaa

augmenter = iaa.SomeOf((0, None),
    [
        iaa.Fliplr(0.5),
        iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 12px (randomly chosen)
        iaa.PerspectiveTransform(scale=(0.01, 0.05), keep_size=True),
        iaa.Affine(rotate=(-20, 20)),
        iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
        iaa.Affine(shear=(-16, 16)),
        iaa.Affine(scale={"x": (0.9, 1.2), "y": (0.9, 1.2)}),
        iaa.GammaContrast((0.5, 2.0)),
        iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
    ],
    random_order=True,
)
