# scGenePredix
A deep learning model employing CNN to predict gene expression from chromatin accessibility counts using a single-cell multiomics dataset.

CS284A AI in Biology & Medicine, Fall 2023

Final Project - Group 4

Imtiyaz Hariyani, Kevin Xu, Faisal Alshaiddi

University of California, Irvine

## **Project Description**
This project introduces a preliminary model for predicting single-cell gene expression using chromatin accessibility counts, specifically focusing on oligodendrocyte cells in the human cerebral cortex across various age groups. Leveraging Convolutional Neural Networks (CNNs), we utilize a published multiomic single-cell dataset (snRNA-seq + snATAC-seq) to train and evaluate the model's predictive performance, utilizing Mean Squared Error (MSE) as a metric. The goal is to decipher gene expression patterns in the developing human brain, with a particular emphasis on the complex relationship between chromatin accessibility and gene expression at the single-cell level.

## **Getting Started**

### Prerequisites
For running the script to train and test the model, make sure you have the following installed:

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

```sh
pip install numpy,pandas,scikit-learn,torch,matplotlib


## **Disclaimer**
The code is still at a preliminary stage and should not be used for the analysis of single-cell data.
