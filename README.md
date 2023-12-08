# scGenePredix
A deep learning model employing CNN to predict gene expression from chromatin accessibility counts using a single-cell multiomics dataset.

CS284A AI in Biology & Medicine, Fall 2023

Final Project - Group 4

Imtiyaz Hariyani, Kevin Xu, Faisal Alshaiddi

University of California, Irvine

## **Project Description**
This project introduces a preliminary model for predicting single-cell gene expression using chromatin accessibility counts, specifically built using the oligodendrocyte cells in the human cerebral cortex across various age groups (infancy, childhood, adolescence, adulthood). Leveraging Convolutional Neural Networks (CNNs), we utilize a published multiomic single-cell dataset (snRNA-seq + snATAC-seq) (1) to train and evaluate the model's predictive performance, utilizing Mean Squared Error (MSE) as a metric. The goal is to decipher gene expression patterns in the developing human brain, with a particular emphasis on the complex relationship between chromatin accessibility and gene expression at the single-cell level.

The model is built exclusively using genes present on the X chromosome to capture spatial patterns, while keeping the dataset to a minimum for the purposes of this class project.

## **Getting Started**

### Prerequisites
For running the script to train and test the model, you need to have the following installed:

1. python (≥3.10)
2. numpy
3. pandas
4. scikit-learn
5. pytorch
6. matplotlib

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/imtiyazhariyani/scGenePredix.git

2. Create a conda environment and install the dependencies:
   ```sh
   conda create -n myenv python=3.10 numpy pandas scikit-learn pytorch matplotlib

3. Activate the conda environment before running the scripts.
   ```sh
   conda activate myenv

Alternatively, you can install dependencies using pip after setting up python version 3.10.    
  
    pip install numpy pandas scikit-learn torch matplotlib

## **Training & Validating the Model**

### Obtaining the Dataset & Preprocessing
To obtain & preprocess the data, run the following script to download the h5ad files, process and obtain the dataset, filter for X chromosome genes only, and add gene coordinates
    
    ./preprocessing.sh

### Training the Model 
If you skipped the previous step to obtain & preprocess the dataset, you can download the final dataset from [here](https://drive.google.com/file/d/1fUkNrLLetrGrObsPIWYBIuZVvR0BfCzH/view?usp=sharing) 

To train the model, run the python script below.

     python main_train_validate.py

## **Testing the Existing Model Trained using the Oligeodendrocyte Cells**

## **Disclaimer**
The code is still at a preliminary stage and should not be used for the analysis of single-cell data.

## **References**
1. Zhu, K., Bendl, J., Rahman, S., Vicari, J. M., Coleman, C., Clarence, T., ... & Roussos, P. (2023). Multi-omic profiling of the developing human cerebral cortex at the single-cell level. Science Advances, 9(41), eadg3754.

