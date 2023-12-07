#VERSION 4 - Updated code to work with paired data (RNA+ATAC) rather than pools of data. This code creates one model per gene to predict gene expression from chromatin accessibility data alongside other features. Still not ideal because we want a single model to be able to learn from other genes but at least now there is some specificity with RNA-ATAC pairs.
#Written by Imtiyaz

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

class PairedCNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.2, num_genes=5):
        super(PairedCNNModel, self).__init__()

        self.input_size = input_size  # Set input size during initialization

        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=0)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.pool2 = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(64, hidden_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_size + 1, 5)

        # Separate branches for each gene
        self.gene_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, 1)
            )
            for _ in range(num_genes)
        ])

    def forward(self, x, a):
        x = x.permute(0, 2, 1)

        x = F.relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        # Separate fully connected layer
        x_fc = F.relu(self.fc1(x))
        x_fc = self.batch_norm3(x_fc)
        x_fc = self.dropout3(x_fc)

        # Separate branches for each gene
        gene_predictions = [gene_branch(x) for gene_branch in self.gene_branches]

        # Concatenate the gene-specific predictions
        x = torch.cat(gene_predictions, dim=1)

        return x.view(x.size(0), -1)

class GeneExpressionPredictor:
    def __init__(self, df):
        self.df = df
        self.model = None
        self.input_size = None
        self.gene_data = {}  # Dictionary to store data for each gene

    def preprocess_data(self):
        # Identify gene expression columns
        gene_expression_columns = self.df.filter(like='_Expression').columns

        # Specify additional features for each cell
        additional_features = ['age_group', 'sex']

        # Use DataFrame to get additional categorical features
        additional_categorical_features = list(set(additional_features) & set(self.df.columns))

        # Fill missing values in specific columns with a default value (e.g., 0)
        default_value = 0
        self.df.fillna(default_value, inplace=True)

        # Identify chromatin accessibility columns
        chromatin_accessibility_columns = self.df.filter(like='_ChromatinAccessibility').columns

        # Combine all feature names (numeric, categorical, and additional features)
        feature_columns = chromatin_accessibility_columns.union(additional_categorical_features)

        # Iterate over each gene to create subsets
        for gene in gene_expression_columns:
            # Select features and target features for the current gene
            X_gene = self.df[feature_columns].copy()
            y_gene = self.df[gene].copy()
            #print(y_gene)

            # Convert categorical columns to one-hot encoding
            X_gene = pd.get_dummies(X_gene, columns=additional_categorical_features)

            # Reshape input data for CNN
            X_gene_tensor = torch.tensor(X_gene.values.reshape(X_gene.shape[0], X_gene.shape[1], 1), dtype=torch.float32)
            y_gene_tensor = torch.tensor(y_gene.values, dtype=torch.float32)

            # Store gene-specific data in the dictionary
            self.gene_data[gene] = {'X_tensor': X_gene_tensor, 'y_tensor': y_gene_tensor}

            print(f"Shapes after reshaping for {gene}:")
            print("X_tensor:", X_gene_tensor.shape)
            print("y_tensor:", y_gene_tensor.shape)

        self.input_size = X_gene.shape[1]


    def train_model(self, hidden_size=5, dropout_rate=0.2, epochs=100, batch_size=32):
        for gene, data in self.gene_data.items():
            X_gene_tensor = data['X_tensor']
            y_gene_tensor = data['y_tensor']

            model = PairedCNNModel(input_size=self.input_size, hidden_size=hidden_size, dropout_rate=dropout_rate)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Store the training and validation loss for plotting
            train_losses = []
            val_losses = []

            for epoch in range(epochs):
                # Training
                model.train()
                optimizer.zero_grad()
                outputs = model(X_gene_tensor, X_gene_tensor[:, -1:, -1:])
                loss = criterion(outputs, y_gene_tensor.unsqueeze(1).expand_as(outputs))
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                # Validation (you can add a validation set if needed)
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_gene_tensor, X_gene_tensor[:, -1:, :])
                    val_loss = criterion(val_outputs, y_gene_tensor.unsqueeze(1).expand_as(val_outputs))
                    val_losses.append(val_loss.item())

                if (epoch + 1) % 10 == 0:
                    print(f'Gene: {gene}, Epoch {epoch+1}/{epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

            # Plot the training and validation loss for each gene
            plt.plot(range(1, epochs+1), train_losses, label=f'Training Loss ({gene})')
            plt.plot(range(1, epochs+1), val_losses, label=f'Validation Loss ({gene})')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        self.model = model

        print("Training complete.")

# Load the data
df = pd.read_csv('sample data - Sheet1.csv')

# Create an instance of GeneExpressionPredictor
gene_predictor = GeneExpressionPredictor(df)

# Preprocess the data
gene_predictor.preprocess_data()

# Train the model
gene_predictor.train_model()

# Evaluate the model
#gene_predictor.evaluate_model()
