import pandas as pd

#Read csv file (output from R script)
x_chromosome_genes_df = pd.read_csv('x_chromosome_genes.csv')

# Read the dataset into a DataFrame
csv_file_path = 'oligodendroyctes.csv'
df = pd.read_csv(csv_file_path, index_col='index')

# Extract the gene IDs from the column names
gene_ids = [col.split('_')[0] for col in df.columns if '_Expression' in col]

# Filter the DataFrame to keep only X chromosome genes
filtered_df = df[df['ensembl_gene_id'].isin(x_chromosome_genes_list)]

# Create new columns for start and end positions
for column in filtered_df.columns:
    # Extract the gene ID from the column name
    gene_id = column.split('_')[0]

    # Filter the x_chromosome_genes_df for the current gene ID
    gene_info = x_chromosome_genes_df[x_chromosome_genes_df['ensembl_gene_id'] == gene_id]

    # Add new columns for start and end positions in filtered_df
    filtered_df[f'{gene_id}_start_position'] = gene_info['start_position'].values[0]
    filtered_df[f'{gene_id}_end_position'] = gene_info['end_position'].values[0]

# Separate gene expression and chromatin accessibility columns
filtered_df = pd.concat([filtered_df.filter(like='Expression'), filtered_df.filter(like='ChromatinAccessibility'), 
                        filtered_df.filter(like='_start_position'), filtered_df.filter(like='_end_position')], axis=1)
# Save the updated DataFrame to a new CSV file
filtered_df.to_csv('finaldataset_oligodendrocytes.csv', index=False)
