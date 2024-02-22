# Airbus Ship Detection Challenge
***
<img width="950" alt="header" src="https://github.com/optimist2point0/Airbus-Ship-Detection-Challenge/assets/62249041/15a040b0-4ed6-4bad-ab9f-8565ff2d4b0d">

## Description
Build a model that detects all ships in satellite images as quickly as possible.

## Solution explanation
The data imbalance was found during data analysis.\
<img width="691" alt="image" src="https://github.com/optimist2point0/Airbus-Ship-Detection-Challenge/assets/62249041/c039a239-7b6b-40df-a3c9-347ae9f75822">

To deal with this issue I split the images without ships and strutify by color gamma (mean of 3 channels).

<img width="971" alt="image" src="https://github.com/optimist2point0/Airbus-Ship-Detection-Challenge/assets/62249041/fc54ed71-e56e-4a52-9551-5a36f740a94e">

***
<img width="967" alt="image" src="https://github.com/optimist2point0/Airbus-Ship-Detection-Challenge/assets/62249041/edde29ab-7fbd-466e-8bfb-37542769baf2">

<img width="964" alt="image" src="https://github.com/optimist2point0/Airbus-Ship-Detection-Challenge/assets/62249041/2562c8e3-e013-451a-8de9-dd797e901a77">

<img width="964" alt="image" src="https://github.com/optimist2point0/Airbus-Ship-Detection-Challenge/assets/62249041/305062a8-0bdd-489f-ba19-92f9b93f7f62">

***
In addition I used simple data augmentation (zoom and flips only). My model has default **U-Net** architecture. For model loss I used $`0.2*binarycrossentropy+ 0.8*(1 - dicecoef)`$ and **Adam** as optimizer. The model was fitting 10 epochs. I reached 0.78 private score by submission on kaggle.
