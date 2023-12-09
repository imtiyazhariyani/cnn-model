#IMTIYAZ VERSION 5 - Testing code on the test dataset using the trained model
#IMTIYAZ VERSION 5 - Training and validating model on final dataset after incorporating start, stop coordinates (oligodendrocytes, all genes on X chr)

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

#comment out if it is already loaded
df = pd.read_csv('final_test.csv')


class PairedCNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.2, num_genes=652):
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

        self.fc1 = nn.Linear(31296, hidden_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Embedding layer for gene identity
        self.embedding = nn.Embedding(num_genes, hidden_size)

        # Linear layers for gene-specific information
        self.fc_gene = nn.Linear(1, hidden_size)
        self.batch_norm_gene = nn.BatchNorm1d(hidden_size)
        self.dropout_gene = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_size + 1, 652)


    def forward(self, x, a):
        x = x.permute(0, 2, 1)

        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.pool2(x)

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.batch_norm3(x)
        x = self.dropout3(x)

        # Process gene-specific information
        a = a.squeeze(dim=-1)
        a_embedding = self.embedding(a.long())
        a_embedding = a_embedding.squeeze(dim=1)
        a_embedding = F.relu(a_embedding)

        # Concatenate shared and gene-specific information
        x = torch.cat([x,a], dim=1)

        # Final linear layer
        x = self.fc2(x)

        return x.view(x.size(0), -1)

class GeneExpressionPredictor:
    def __init__(self, df):
        self.df = df
        self.X_train_tensor = None
        self.y_train_tensor = None
        self.X_val_tensor = None
        self.y_val_tensor = None
        self.X_test_tensor = None
        self.y_test_tensor = None

        self.weight_tensor = None #weight

    def preprocess_data(self):

        # Specify additional features for each cell
        # Identify columns that contain the words "age_group" or "sex"
        selected_feature_columns = self.df.filter(like='age_group').columns.union(self.df.filter(like='sex').columns)

        # Use the selected columns as additional features
        additional_categorical_features = list(set(selected_feature_columns))

        # Identify chromatin accessibility columns
        chromatin_accessibility_columns = self.df.filter(like='_ChromatinAccessibility').columns

        # Identify gene start columns
        start_columns = self.df.filter(like='_start_position').columns
        
        # Identify gene stop columns
        stop_columns = self.df.filter(like='_end_position').columns

        # Combine all feature names (numeric, categorical, and additional features)
        feature_columns = (chromatin_accessibility_columns.union(additional_categorical_features).union(start_columns).union(stop_columns)
)

        # Select features and target features
        X_test = self.df[feature_columns].copy()
        y_test = self.df.filter(like='_Expression').copy()

        # Reshape data
        self.X_test_tensor = torch.tensor(X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1), dtype=torch.float32)
        self.y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    def train_model(self, hidden_size=128, dropout_rate=0.2, epochs=100):
        print("Welcome. Using scGenePredix to train the model.")
        input_size = self.X_train_tensor.shape[1]

        model = PairedCNNModel(input_size=input_size, hidden_size=hidden_size, dropout_rate=dropout_rate)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Store the training and validation loss for plotting
        train_losses = []
        val_losses = []

        print("Training in progress...")

        for epoch in range(epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            outputs = model(self.X_train_tensor, self.X_train_tensor[:, -1:, -1:])
            #loss = criterion(outputs, self.y_train_tensor)
            loss = torch.mean(self.weight_tensor * (outputs - self.y_train_tensor) ** 2) #loss with weights
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(self.X_val_tensor, self.X_val_tensor[:, -1:, -1:])
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
        plt.ylim(0, 0.5)  # Adjust y-axis limits
        plt.show()

        print("Training complete.")
        torch.save(model.state_dict(), 'trained_model_oligodendrocytes.pth')
        print("Trained model saved to 'trained_model_oligodendrocytes.pth'")
        self.model = model

    def load_model(self, model_path='trained_model_oligodendrocytes.pth'):
        # Pass the input_size to the model during initialization
        input_size = 31296
        self.model = PairedCNNModel(input_size=input_size, hidden_size=128, dropout_rate=0.2)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print(f"Model loaded from {model_path}")

    def evaluate_model(self):

        model = self.model.eval()
        criterion = nn.MSELoss()

        # Evaluate on the test set
        with torch.no_grad():
            test_outputs = model(self.X_test_tensor, self.X_test_tensor[:, -1:, :])
            test_loss = criterion(test_outputs, self.y_test_tensor)

        print(f'Test Loss: {test_loss.item()}')

        return test_loss.item()


# Create an instance of GeneExpressionPredictor
gene_predictor = GeneExpressionPredictor(df)

# Preprocess the data
gene_predictor.preprocess_data()

# Train the model
#gene_predictor.train_model()

# Load the model
gene_predictor.load_model()

# Evaluate the model
gene_predictor.evaluate_model()
