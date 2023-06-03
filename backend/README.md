### Data Format

The result of the pipeline will be returned as a dictionary of image_path : feature_vector:

```
    {
        "img1.png" : (#, #, #, #),
        "img2.png" : (#, #, #, #),
        "img3.png" : (#, #, #, #),
        ...
    }
```

### Classifier input format

The result of the feature extraction should return a feature vector for every cropped image. For training we give Marvin a list of feautures together with class labels ('pet', 'can', 'glass').