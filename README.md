# Introduction

This repo is the official implementation of our paper: [Extraction of cropland field parcels with High-Resolution Remote Sensing Images using multi-task learning](https://www.tandfonline.com/doi/full/10.1080/22797254.2023.2181874). The prediction process is shown in figure below.

- ![avatar](./images/flowchart.png)

# Data and checkpoint

Google cloud: https://drive.google.com/drive/folders/1f1jZwUCS4892bkne7Ob1iWmcFhj8aDIZ?usp=sharing

# Denmark datastet
link：https://pan.baidu.com/s/1l9W0ekFdvyj8FhKA44CNuQ 
pw：qazj 

# How to use

1. To obtain the cropland parcel(without using patch refinement)

   ```bash
   python main.py --img ${image_path} --weights ${weights_path} --shp ${the_path_of_output_in_shapefile}
   ```

   

2. The next step is repair the break line

   ```bash
   python PatchRefinement.py --lineDN ${lineDN_path} --img ${image_path} --weights ${weights_path}
   ```
   PS:The training code can be found [here](https://github.com/carlesventura/iterative-deep-learning).
   

3. In final, merge the results of the two steps.

   ```bash
   python postprocess.py --img ${image_path} --shp ${the_path_of_output_in_shapefile}
   ```

# Training code
The code works a bit cluttered and without comments, download it if you need.
link：https://pan.baidu.com/s/1AKwMW9-0t8MA6LoEzgt6VA?pwd=kqau 
pwd：kqau 
 
