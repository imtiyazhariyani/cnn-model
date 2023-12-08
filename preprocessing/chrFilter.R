# Install and load the biomaRt package
if (!requireNamespace("biomaRt", quietly = TRUE)) {
  install.packages("biomaRt")
}
library(biomaRt)

# Connect to the Ensembl BioMart database
ensembl <- useMart("ENSEMBL_MART_ENSEMBL", host = "www.ensembl.org", path = "/biomart/martservice", dataset = "hsapiens_gene_ensembl")
ensembl_ids <- c("ENSG00000146938")
# Define the list of Ensembl IDs
ensembl_ids <- read.csv("all-genes.csv", sep="")

# Get the attributes you want (e.g., gene ID, chromosome name)
attributes <- c("ensembl_gene_id", "chromosome_name", "start_position", "end_position")

# Query BioMart
results <- getBM(attributes = attributes, filters = "ensembl_gene_id", values = ensembl_ids, mart = ensembl)

# Filter results to keep genes found on the X chromosome
x_chromosome_genes <- subset(results, chromosome_name == "X")

# Save the X chromosome genes to a CSV file
write.csv(x_chromosome_genes, file = "x_chromosome_genes.csv", row.names = FALSE)