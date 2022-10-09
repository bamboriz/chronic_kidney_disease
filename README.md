# Biomedical Data Science - Chronic Kidney Disease
This is a proof of concept that can help physicians better understand chronic kidney disease (CKD) using numerous measurements and biomarkers that have been collected. The dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease).

## Installation
To run this locally, you need Python 3.6 or greater installed. It is advisable to create a separate environment before running the following commands
```sh
$ git clone [url]
$ cd /chronic_kidney_disease
$ pip install -r requirements.txt
$ streamlit run "./app/Risk _Factors_for _CKD.py"
```
> Note: This project has been fully parameterised to work with similar datasets in csv format. However if your dataset needs specific preprocessing, please do so and rename your dataset into data.csv which must be placed in the **data** folder.

For example the provided dataset comes in _.arrf_ formart. Feel free to run the **prepare_dataset.py** file to generate a **data.csv** file.
```sh
python app/prepare_dataset.py ./data/chronic_kidney_disease_full.arff
```
## Overview
The Proof of Concept has 2 main parts. 

The first section allows for exploring the dataset by looking at the characteristics of various features, their relationship with each other as well as in relation to the target variable.

<img src="/assets/scatter.png" width="300">

In the second section, a physician can deep-dive by filtering the dataset on a specific value of the target variable. This allows for identification of different disease subtypes with the help of clustering techniques.

<img src="/assets/cluster.png" width="300">

## Running Tests
```sh
pytest -v
```