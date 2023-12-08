#Preprocessing of Data
#Written by Imtiyaz

#Part 1 - Download data from https://cellxgene.cziscience.com/collections/ceb895f4-ff9f-403a-b7c3-187a9657ac2c 
#snRNA-seq:
wget https://datasets.cellxgene.cziscience.com/1e244d0d-b3d3-419c-8e2e-ed8456ba471c.h5ad

#snATAC-seq: 
wget https://datasets.cellxgene.cziscience.com/d5adf2ac-95f8-48df-b6f8-674d44df6529.h5ad

#Part 2 - Use Scanpy to process the data and obtain a dataset for model training & evaluation

python scanpy_process.py

#Part 3 - Run the R script chrFilter.R to obtain a list of genes on the X chromosome

R chrFilter.R

#Part 4 - Add coordinates to the dataset csv file

python add_coordinates.py


#Part 5 - Select 5 genes for testing the model and use the entire dataset for training the model