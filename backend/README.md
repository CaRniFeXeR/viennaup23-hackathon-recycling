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