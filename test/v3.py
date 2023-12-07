#VERSION 3 - Fixed issues with input features including chromatin accessibility which was not included and gene coordinates; removed scaling since data is normalized; fixed CNN input dimensions, ran the model on test dataset for 5 genes, 1000 epochs
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

class CNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.2):
        super(CNNModel, self).__init__()

        self.input_size = input_size  # Set input size during initialization

        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=0)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.pool2 = nn.MaxPool1d(2)

        # Calculate the size of the linear layer input based on the output size after convolutions
        # conv_output_size = 64 * (((input_size - 2) // 2 - 2) // 2 - 2)
        self.fc1 = nn.Linear(64, hidden_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_size + 1, 5)

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

        x = F.relu(self.fc1(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)

        a = a.squeeze(dim=-1)

        #print(x.size(-2),x.size(-1),x.size(0),x.size(1))
        #print(a.size(-2),a.size(-1),a.size(0),a.size(1))

        # Ensure that the size of dimension 1 is consistent


        x = torch.cat([x, a], dim=1)

        x = self.fc2(x)
        return x.view(x.size(0),-1)

class GeneExpressionPredictor:
    def __init__(self, df):
        self.df = df
        self.model = None
        self.input_size = None
        self.X_train_tensor = None
        self.y_train_tensor = None
        self.X_val_tensor = None
        self.y_val_tensor = None
        self.X_test_tensor = None
        self.y_test_tensor = None

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

        # Select features and target features
        X = self.df[feature_columns].copy()
        y = self.df[gene_expression_columns].copy()

        # Convert categorical columns to one-hot encoding
        X = pd.get_dummies(X, columns=additional_categorical_features)

        # Train-validation-test split
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

        # Reshape input data for CNN
        self.X_train_tensor = torch.tensor(X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1), dtype=torch.float32)
        self.y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

        self.X_val_tensor = torch.tensor(X_val.values.reshape(X_val.shape[0], X_val.shape[1], 1), dtype=torch.float32)
        self.y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

        self.X_test_tensor = torch.tensor(X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1), dtype=torch.float32)
        self.y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

        print("Shapes after reshaping:")
        print("X_train_tensor:", self.X_train_tensor.shape)
        print("y_train_tensor:", self.y_train_tensor.shape)
        print("X_val_tensor:", self.X_val_tensor.shape)
        print("y_val_tensor:", self.y_val_tensor.shape)
        print("X_test_tensor:", self.X_test_tensor.shape)
        print("y_test_tensor:", self.y_test_tensor.shape)

        self.input_size = X.shape[1]


    def train_model(self, hidden_size=5, dropout_rate=0.2, epochs=1000, batch_size=32):
        model = CNNModel(input_size=self.input_size, hidden_size=hidden_size, dropout_rate=dropout_rate)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Store the training and validation loss for plotting
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            outputs = model(self.X_train_tensor, self.X_train_tensor[:, -1:, :])
            loss = criterion(outputs, self.y_train_tensor)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(self.X_val_tensor, self.X_val_tensor[:, -1:, :])
                val_loss = criterion(val_outputs, self.y_val_tensor)
                val_losses.append(val_loss.item())

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

        # Plot the training and validation loss
        plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
        plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        self.model = model

    def evaluate_model(self):
        if self.model is None:
            print("Model not trained. Please train the model before evaluation.")
            return

        model = self.model.eval()
        criterion = nn.MSELoss()

        # Evaluate on the test set
        with torch.no_grad():
            test_outputs = model(self.X_test_tensor, self.X_test_tensor[:, -1:, :])
            test_loss = criterion(test_outputs, self.y_test_tensor)

        print(f'Test Loss: {test_loss.item()}')



# Load the data
df = pd.read_csv('sample data - Sheet1.csv')

# Create an instance of GeneExpressionPredictor
gene_predictor = GeneExpressionPredictor(df)

# Preprocess the data
gene_predictor.preprocess_data()

# Train the model
gene_predictor.train_model()

# Evaluate the model
gene_predictor.evaluate_model()
