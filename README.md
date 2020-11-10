# Polygon generator

* The algorithm starts with the random point, then creates the polygon by sampling points on a circles centered at the point. 
* Randon variance is added by random angles between sequential points and random radii from the centre to each point.
* Additional verifications are performed to ensure simple polygons and no collinearity between adjacent vertex.
* Images are generated with random nubmer of the following distortions:
>    GaussianBlur(sigma=(0.0, 4.0)),
>    MotionBlur(k=15),
>   MultiplyElementwise((0.5, 1.5), per_channel=0.5), # Noise inside figure
>    Multiply((0.5, 1.5)), #Brightness
>    Dropout(p=(0, 0.2)), #Random pxls to 0
>    CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)), #Big rectangular dropout
>    BlendAlphaSimplexNoise(iaa.EdgeDetect(1.0), sigmoid_thresh=iap.Normal(10.0, 5.0)), #Blurred blobs
>    BlendAlphaHorizontalLinearGradient(iaa.AveragePooling(11), start_at=(0.0, 1.0), end_at=(0.0, 1.0)), #Massive blend
>    PiecewiseAffine(scale=(0.01, 0.02)), #Little curve
>    AveragePooling([2, 8])

* Under defualt setting, generation of 1k images takes 14 sec.
* Returns a list of vertices, in CCW order. In most cases, first point will be middle left one.

# Config:

|Parameter | Description|
|---------------------|-------------------------------------------|
|img_num: | how many images to generate|
| distort:  | whether to apply image distortion|
| numVerts:  | number of polygon vertex|
| dir:       | dir containing 'images" folder|
| collin_coef: | [0,1] how far angles shall be from 180°|
| aveRadius: | mean for sampling radii|
| angular_variance: | [0,1] the variance in angles|
| radii_variance: | [0,1] the variance in length|
| input_size: | image size will be (input_size,input_size)|

# Requirements:

For image distortion latest ```imgaug``` is required:

```!pip install git+https://github.com/aleju/imgaug.git```

Also usual stuff: python-openCV, numpy, pandas

# Project structure

> poly_gen.py

> poly_gen_config.yaml

> dir

>>images
