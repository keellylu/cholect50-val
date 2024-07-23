### NAME: CholecT50
### WEBSITE: http://camma.u-strasbg.fr/datasets
### GITHUB: https://github.com/CAMMA-public/cholect50
### DATSET PUBLICATION: https://arxiv.org/abs/2109.03223
### OFFICIAL DATA SPLIT: https://arxiv.org/abs/2204.05235
### LICENSE: Release for non-commercial scientific research purposes protected by CC BY-NC-SA 4.0 LICENSE


# Description
The data consists of endoscopic videos obtained during laparoscopic cholecystectomy surgeries at the University Hospital of Strasbourg, France. The CholecT50 dataset consists of 50 videos captured by endoscope during the surgeries and annotated with triplet information about surgical actions in the videos. A triplet is in the form of <instrument, verb, target>. The phase labels are also provided. Spatial annotations in the form of bounding boxes over the instrument tips are provided for 5 videos. The box-triplet matching labels are also provided for all bounding box annotations. All surgeries are annotated frame-wise by expert surgeons. To ensure anonymity, frames corresponding to extra-abdominal views are censored by entirely black (RGB 0 0 0) frames. CholecT50 is a superset of CholecT40 and CholecT45. 


# Usage
The CholecT50 dataset can support the following research:

- Surgical action triplet recognition
- Surgical action triplet detection/localization
- Surgical tool presence detection
- Surgical tool detection/localization
- Surgical action/verb recognition
- Surgical target recognition
- Surgical phase recognition

And any combination of the above


# Current Release 2.0
The latest release contains 50 videos with binary presence labels for the:
- triplets
- instruments
- verbs/actions
- targets/anatomies
- phases

>> Bounding box labels will be available in the next release 3.0.

The statistics of the dataset are provided in the reference publication as well as on the GitHub repository of the dataset.


# Reference Publication
This dataset could only be generated thanks to the continuous support from our surgical partners. In order to properly credit the authors and clinicians for their efforts, you are kindly requested to cite the work that led to the generation of this dataset:

C.I. Nwoye, T. Yu, C. Gonzalez, B. Seeliger, P. Mascagni, D. Mutter, J. Marescaux, N. Padoy. Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos. Medical Image Analysis, 78 (2022) 102433.


# Data Structure and Format
Please, visit our GitHub page for details on the data structure and format.


# Data Split and versions
Check the [paper](https://arxiv.org/abs/2204.05235) for the official split of the dataset. 
The paper also provides:

- details on the versions of the dataset
- recommended usage pattern
- recommended evaluation metrics for research
- benchmark results of some baseline models on the different dataset versions. The up-to-date leaderboard on model performances on the dataset is provided on the GitHub page.


# Data loader
TensorFlow and PyTorch implementations for loading the dataset are provided on the GitHub repository.


# Benchmark Challenge
Presently, two Endoscopic vision challenge has been conducted on the dataset. Their results are summarized in the papers:

- Nwoye C.I et al., CholecTriplet2021: A benchmark challenge for surgical action triplet recognition. Medical Image Analysis 2022. arXiv [link](https://arxiv.org/abs/2204.04746)
- Nwoye C.I et al., CholecTriplet2022: Show me a tool and tell me the triplet -- an endoscopic vision challenge for surgical action triplet detection. arXiv [link](https://arxiv.org/abs/2302.06294)



# Video Overlap
Since the dataset is from the CAMMA research group, at the University of Strasbourg, France, there are possible video overlaps in other cholecystectomy datasets such as Cholec80, Cholec120, M2CAI16, etc. The video IDs (e.g. 1, 2, 5, 80, etc.) are consistent across these datasets. The prefix "VID" in the video filenames (e.g. VID01, VID02, VID80, etc.) are sometimes written as "Video" in other datasets (e.g. Video01, Video80, etc. ). Researchers are advised to take into consideration the overlapping videos when pre-training their models on other cholecystectomy datasets.



*For more information visit the GitHub page of the dataset.*

------------------------------------------------------------

 
Chinedu Nwoye, PhD
For CAMMA, ICube, University of Strasbourg, France

