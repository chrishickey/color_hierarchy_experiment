# Experiment 1 
This is the code that was used for running the 'CNN Color Classification Recall Experiment' in the Color Hierarchy study.

NOTE: 

The model used in this experiment is a modified version of the model taken from: https://github.com/beerboaa/Color-Classification-CNN

The R code used to run ANOVA analysis is a modified version of the code from (a previous paper I co-authored): https://github.com/sungjae-cho/arithmetic-jordan-net

## Install requirements
To install all requirements necessary for this experiment run
```bash 
pip install -i requirements.txt
```
## Get data
Unfortunately the zip file of all the images is too large to upload onto github. 
Please send me an email if you would like the dataset used in this experiment.

## Run Experiment 1
To start the experiment, run
```bash 
python colour_class.py --run_experiment --train_dir 'data/training_data' --test_dir 'data/val_data' --results_dir opp_results --transform opp
```
where the arguments are as follows

| Argument   | Meaning  |
|---|---|
| --run_experiment | Runs the experiment |
| --train_dir | Specifies the dir where training data is located |
| --test_dir  | Specifies the dir where val data is located |
| --results_dir | Specifies the dir where results will be stored |
| --transform | Specifies color space to use (opp, rgb, bgr, ybr, yuv)|  
| --num_models | (Optional), the number of models to run |  

For each "num_model" number of models run in this experiment, the order in which they are learned
will be written to the results directory.

## Process experiment 1 results
To process results of the experiment, run
```bash 
python colour_class.py --process_results --results_dir 'results'
```
where the arguments are as follows

| Argument   | Meaning  |
|---|---|
| --process_results | Process results |
| --results_dir | Should be the same results dir as set in the experiment |
 
 This will output two files. One of these files will end in 'Rprocessible.csv'. In
 order to run ANOVA analysis on this file, follow the comments outlined in anova.R.
 Note that several 100 models are required in order to get normally distributed samples of all
 data sample points for all colors, suitable for ANOVA analysis.
 
 Note:
 
 The results sighted in the paper are all included in the "results" directory. To recreate the
 statistics cite in the paper, please run the anova.R file after uncommenting the relevant 
 Rprocessible results file.

## Hierarchical Color Learning differences
Average number of epochs taken to learn each color for each color space ("learning is defined as consistent recall > .85")

Results from this experiment (as can be reproduced from results in the results directory) are as follows

#### Average Epochs to Learn colors

|Color Space | Red | Yellow | Green | Purple | Blue | Brown | Orange | Gray |
|---|---|---|---|---|---|---|---|---|
|OPP | 19.44 | 23.50 | 24.73 | 24.55 | 28.38 | 33.38 | 34.01 | 33.08 |
|RGB | 22.99 | 27.73 | 27.09 | 29.20 | 31.99 | 36.60 | 37.43 | 36.75 |
|BGR | 22.62 | 27.12 | 27.15 | 28.90 | 31.39 | 36.17 | 37.23 | 36.30 |
|YCbCr | 17.49 | 23.80 | 26.78 | 20.49 | 23.06 | 33.44 | 33.58 | 32.55 |
|YUV | 19.62 | 22.69 | 25.72 | 20.56 | 21.60 | 32.72 | 32.75 | 31.92 |

#### Statistically significant differences between colors (ANOVA)

| CS | F | Epochs to Learn |
|---|---|---|
|OPP |206 | red < yellow = green = purple < blue < brown = gray = orange |
|RGB |141 |  red < yellow = green = purple < blue < brown = gray = orange |
|BGR | 151 | red < yellow = green = purple < blue < brown = gray = orange | 
|YUV | 232 | red = purple = blue < yellow < green < brown = gray = orange | 
|YCbCr | 270 | red < purple < yellow = blue < green < brown = gray = orange |
