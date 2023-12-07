#VERSION 2 - Included Start & Stop coordinates for genes, class weights, ensured equal distribution of age classes across training and test data
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
from tensorflow.keras.regularizers import l2

# Load the data
df = pd.read_csv('sample data - Sheet1.csv')

# Identify all genes by looking for unique identifiers in the column names
genes = set(col.split('_')[0] for col in df.columns if '_Start' in col or '_Stop' in col or '_End' in col)

# For each gene, calculate the length as Stop - Start
for gene in genes:
    start_col = f'{gene}_Start'
    stop_col = f'{gene}_Stop' if f'{gene}_Stop' in df.columns else f'{gene}_End'

    # Ensure the start and stop columns are numeric
    df[start_col] = pd.to_numeric(df[start_col], errors='coerce')
    df[stop_col] = pd.to_numeric(df[stop_col], errors='coerce')

    # Create a new column for gene length
    length_col = f'{gene}_Length'
    df[length_col] = df[stop_col] - df[start_col]

# Fill missing values in specific columns with a default value (e.g., 0)
default_value = 0
df.fillna(default_value, inplace=True)

# Identify categorical and numeric columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.union([f'{gene}_Length' for gene in genes])

# Create a ColumnTransformer to transform categorical columns and scale numeric columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), list(numeric_cols)),
        ('cat', OneHotEncoder(), list(categorical_cols))
    ])

# Fit and transform the ColumnTransformer on the DataFrame
df_transformed = preprocessor.fit_transform(df)

# Collect feature names from the ColumnTransformer
numeric_features = preprocessor.named_transformers_['num'].get_feature_names_out()
categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out()

# Combine all feature names (numeric and categorical)
all_feature_names = list(numeric_features) + list(categorical_features)

# Convert the transformed array back to a DataFrame
df_transformed = pd.DataFrame(df_transformed, columns=all_feature_names)

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
#X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=df['age_group'])

# Reshape input data for CNN
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Create CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1), kernel_regularizer=l2(0.01)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(len(gene_expressions), kernel_regularizer=l2(0.01)))  # Output layer for gene expressions

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_data=(X_test_reshaped, y_test))

# Evaluate model
loss = model.evaluate(X_test_reshaped, y_test)
print(f'Model Loss: {loss}')