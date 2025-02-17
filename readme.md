# CRM measurements using automated Facial Keypoint Tracking
<!-- Code for the paper "Consistent Movement of Viewers' Facial Keypoints While Watching Emotionally Evocative Videos"  -->

## Installation
The Facial Landmark Frontalization algorithm [paper](https://ieeexplore.ieee.org/document/9190989) [code](https://github.com/Shivansh-ct/Consistent-Keypoints/tree/main/FacialLandmarkFrontalization_66) is used in preprocessing.   
## Dataset
The DISFA dataset can be obtained at [DISFA](http://mohammadmahoor.com/disfa/).
## Code Usage
**KPM_data_generation.ipynb** is used for preprocessing and final KPM data generation from the dataset.  
### AU consistency
**au_coding_combine.py** is a helper file to generate AU consistency across the video timeline using **gen_AU_consistency.ipynb**.   
**analysis_AU_consistency.ipynb** is used to analyze CRM in the DISFA dataset using the **AU consistency** metric.
### Keypoint based metrics
**gen_keypoint_metrics.ipynb** is used to generate all the keypoint-based metrics across the video timeline.

### AU consistency vs. Keypoint-based metrics
**compare_all_metrics.ipynb** is used to evaluate the comparison between AU consistency and Keypoint-based metrics.




<!-- # Contents:
```tree
├── PCA_train                                      [Directory: PCA training, visualizations etc.]
│   ├── helper.py                   [functions for feature generation, metrics and face morphing] 
│   ├── pca_train.py                        [Training the PCA on any dataset]
│   ├── pca_train_report.py                                [Generates the train-test table at 95% Train VE]
│   ├── pca_train_report_plot.ipynb  [Plots the train-test plot of variance explained]
│   ├── pca_train_report_plot.py  [compile the metrics performance for train and test from one to total components]
│   ├── pca_train_report_statistics.ipynb  [generates p-value for the significance test]
│   └── pca_train_report_statistics.py       [used for significance test]
└── PCA_vs_FACS                                        [Directory: compares PCA AUs, pure AUs, and comb AUs]
│   ├── choldelete.m                    [helper function for LARSEN algorithm]
│   ├── cholinsert.m              [helper function for LARSEN algorithm]
│   ├── larsen.m                [LARSEN algorithm to generate encodings] 
│   ├── pca_vs_facs_plot.ipynb   [plot graphs for comparing PCA AUs, pure AUs, and comb AUs] 
│   └── pca_vs_facs_plot.m                               [compile the metrics performance for all AUs from one to total components] 
│   data_and_feature_creation.ipynb                                  [code to generate features from facial keypoints]
│   matlab_versions.txt                [required version for matlab]
└── python_versions.txt                [required version of libraries in python] -->