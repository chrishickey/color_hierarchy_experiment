# Experiment 2
This is the code that was used for running the 'CNN Color Object Detection Experiment' in Color Hierarchy given study.

NOTE: 

Much of the Faster RCNN code used in this project was taken and modified from the official pytorch
website and associated github pages. Additionally some code from the cocoapi is also included in
this repo.

The code for calculating recall scores was gotten by editing source code taken from : 
https://github.com/Cartucho/mAP

## Install requirements
To install all requirements necessary for this experiment run
```bash 
pip install -i requirements2.txt
```
## Get data
The dataset used in this experiment was the modanet dataset:
https://github.com/eBay/modanet

Follow instructions on this github page to get the data.

## Unzip annotations 
Train and test color annotations for this experiment are located in 
the 'color_annotations.zip' file. These files will be required for running the 
experiment so you must first unzip this file.
```bash 
unzip color_annotations.zip
```

## Run Experiment 2
To start the experiment, run
```bash 
python run_experiment.py --color_space OPP --images_dir 'modanet_dir/images' 
    --train_annotations 'color_annotations/train' 
    --test_annotations 'color_annotations/test' --identifier OPP_FIRST_STUDY
```
where the arguments are as follows

| Argument   | Meaning  |
|---|---|
| --color_space | Specified color space in which to pass images to the Faster RCNN (OPP, YBR, YUV, RGB) |
| --images_dir | Specified directory in which all modanet images are stored (both training and test images should be all mixed together in one directory) |
| --train_annotations  | Specifies directory where the annotations for the training dataset are located |
| --test_annotations | Specifies directory where the annotations for the test dataset are located |
| --identifier | Specifies a unique identifier which will be added to all results files produced by this execution of the experiment|  

For every iteration of this experiment, results files will be produced under the
'mAP_recall' folder. The results file titled <color_space>_<identifier>_<epoch_num>.txt
will give information on mAP scores on the modanet images specified by the --test-annotations argument for that epoch. The results
file titled  <color_space>_<identifier>_<epoch_num>.json provides information on recall values
per color (Red, Green, Yellow, Blue, Brown, Pink, Purple, Orange, Gray) and adjective used to describe color
(Vivid, Strong, Deep, Light, Brilliant, Moderate, Dark, Pale). Model weights files will also be saved for each epoch.

## Process experiment 2 results
To process how results colors and adjectives were learned for any epoch of the Faster RCNN, 
first pick the corresponding JSON file from the mAP_recall directory. For example, if you want
to see how colors and adjectives were learned for the 10th epoch of the OPP space from the execution identified
by the OPP_FIRST_STUDY identifier, the results file you want will be OPP_OPP_FIRST_STUDY_10.json. Add this file_name as 
a string to the top of the process_python.py file as directed by the comment on the top of that file.
Next run;
```bash 
python process_python.py
```

 Note:
 
The results sighted in the paper are all included in the 'results/results_mAP69.2.json' file. 
To recreate the statistics cited in the paper, please run the process_results.py file with JSON_FILE = 'results/results_mAP69.2.json'
uncommented at the top of the process_results.py file.

Results from this experiment (as can be reproduced from the process_results.py file) are as follows

#### Recall per color category


| Category | Red | Green | Blue | Purple | Yellow | Pink | Brown | Orange | Gray|
|---|---|---|---|---|---|---|---|---|---|
| Outer | .725 | .680 | .669 | .667 | .647 | .557 | .571 | .650 | .696 |
| Skirt | .819 | .673 | .732 | .727 | - | .62 | .654 | - | - |
| Bag | .752 | .646 | .669 | .656 | .675 | .66 | .694 | .702 | - |
| Footwear | .807 | .784 | .805 | .724 | .752 | .621 | .740 | .698 | .755 |
| Belt | .657 | - | .456 | .517 | - | .462 | .660 | .481 | - |
| Top | .629 | .661 | .614 | .632 | .619 | .736 | .580 | .626 | .423 |
| Dress | .702 | .718 | .698 | .690 | - | .660 | .500 | - | - |
| Pants | .914 | .859 | .911 | .878 | - | .830 | .713 | - | - |
| Mean | .751 | .717 | .694 | .686 | .673 | .643 | .639 | .631 | .625 |


#### Recall per descriptive adjective

| Category |Brilliant | Vivid |Deep | Strong | Dark | Moderate | Light | Pale |
|---|---|---|---|---|---|---|---|---|
| Outer | .725 | .671 | .703 | .614 | .715 | .564 | .602 | .663 |
| Skirt | .805 | .813 | .758 | .767 | .739 | .652 | .633 | .610 |
| Boots | - | - | .556 | .559 | .497 | .421 | .364 | .265 |
| Bag | .790 | .742 | .710 | .716 | .675 | .663 | .632 | .636 |
| Footwear | .807  | .789 | .755 | .774 | .73 | .718 | .733 | .697 |
| Belt | - | .563 | .639 | .599 | .577 | .561 | .502 | .402 |
| Top | .722 |.722 | .621 | .665 | .575 | .579 | .644 | .654 |
| Dress | .720 | .726 | .636 | .697 | .664 | .647 | .659 | .663 | 
| Pants | - | .902 | .872 | .831 | .857 | .815 | .845 | .843 |
| Scarf | - | .296 | .390 | .351 | .330 | .382 | .305 | .303 |
| Shorts | - | .759 | .709 | .763 | .707 | .795 | .72 | .778 |
| Headwear | - | - | .703 | - | .711 | .641 | .686 | .617 |
| Mean | .762 | .698 | .671 | .667 | .648 | .620 | .610 |  .594 |

