import anndata
import pandas as pd

# Load the ATAC and RNA AnnData files
atac_file = "scatac.h5ad"
rna_file = "scrna.h5ad"

adata_atac = anndata.read_h5ad(atac_file)
adata_rna = anndata.read_h5ad(rna_file)

# Assuming you have a cell type annotation column in both datasets
cell_type_to_filter = "Oligodendrocytes"

# Filter cells based on cell type
filtered_cells_atac = adata_atac[adata_atac.obs['author_cell_type'] == cell_type_to_filter].copy()
filtered_cells_rna = adata_rna[adata_rna.obs['author_cell_type'] == cell_type_to_filter].copy()

result_df = pd.DataFrame({'index': filtered_cells_atac.obs['index'], 'author_cell_type': filtered_cells_atac.obs['author_cell_type'], 'age_group': filtered_cells_atac.obs['age_group'], 'sex': filtered_cells_atac.obs['sex']})

# Select the relevant columns for the CSV file
#columns_to_select = ['CellID', 'CellType_x']  # Adjust based on your actual column names

# Extract all gene names from the ATAC dataset
all_genes = adata_atac.var_names

# Add columns for gene expression and chromatin accessibility counts
for gene in all_genes:
    # Check if the gene is present in both RNA and ATAC datasets
    if gene in filtered_cells_rna.var_names and gene in filtered_cells_atac.var_names:
        gene_exp_rna = filtered_cells_rna[:, gene].X.toarray().flatten()
        gene_access_atac = filtered_cells_atac[:, gene].X.toarray().flatten()

        # Add gene expression and chromatin accessibility counts to the DataFrame
        result_df[f'{gene}_Expression'] = gene_exp_rna
        result_df[f'{gene}_ChromatinAccessibility'] = gene_access_atac


# Create the final DataFrame

# Save the DataFrame to a CSV file
csv_output_file = "oligodendrocytes.csv"
result_df.to_csv(csv_output_file, index=False)