#VERSION 1 - Preliminary code used for group presentation and training the first model
#Written by Kevin

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D

df = pd.read_csv('sample data - Sheet1.csv')

# Identify categorical and numeric columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Create a ColumnTransformer to transform categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Fit and transform the ColumnTransformer on the DataFrame
df_transformed = preprocessor.fit_transform(df)

# Update feature names after one-hot encoding (for older scikit-learn versions)
feature_names = list(numeric_cols)
for col in categorical_cols:
    unique_values = df[col].dropna().unique()
    feature_names.extend([f'{col}_{val}' for val in unique_values])

# Convert the transformed array back to a DataFrame
df_transformed = pd.DataFrame(df_transformed, columns=feature_names)

# Specify gene expression columns
gene_expressions = ['ENSG00000146938_Expression', 'ENSG00000101849_Expression', 'ENSG00000047644_Expression', 'ENSG00000073464_Expression', 'ENSG00000101871_Expression']

# Select features and targets
feature_columns = df_transformed.columns.difference(gene_expressions)
X = df_transformed[feature_columns]
y = df_transformed[gene_expressions]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape input data for CNN
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Create CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(5))  # Output layer for 5 gene expressions

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_data=(X_test_reshaped, y_test))

# Evaluate model
loss = model.evaluate(X_test_reshaped, y_test)
print(f'Model Loss: {loss}')