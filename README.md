# Face-detection-with-Tensorflow-object-detection-api

The goal of this repo is to make a tool to quickly and easily de-identify a large number of images that is free to use even for commercial use.

This takes a directory of images and removes the faces from the images. It uses the Tensorflow Object detection API and used a pretrained faster_rcnn_resnet101 model that was retrained using hand-labeled images that were marked for commercial use. 

In the future, I would also like it to block out or remove images from the output that have tattoos but I was not successful in my initial attempt. 

To use this tool install the Tensorflow Object detection API, clone the repo and run the image_detect.py followed by the relative path to the directory for the input images and a relative path for the output images that have the faces removed. The defaults are `input/` and `output/`
```
python3 image_detect.py --input_images_path=input --output_images_path=output/
```

Here are some examples of it in action!
![alt text](https://raw.githubusercontent.com/john-cusack/Face-detection-with-Tensorflow-object-detection-api/master/deid_v1.png)
