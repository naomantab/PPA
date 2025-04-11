from Bio import SeqIO
from collections import defaultdict

# Input FASTA file
input_fasta = r"C:\Users\tnaom\OneDrive\Desktop\PPA\01_input_data\reference_seq\GCF_000001635.27_GRCm39_rna.fna"

# Dictionary to count isoforms per gene
gene_isoform_count = defaultdict(int)

# Read FASTA file
for record in SeqIO.parse(input_fasta, "fasta"):
    header_parts = record.description.split()
    gene_name = None

    # Extract gene name (assuming it's in parentheses)
    for part in header_parts:
        if "(" in part and ")" in part:  # Looks for (Zfp85) in description
            gene_name = part.strip("()")  # Remove parentheses
            break

    # Count occurrences of each gene
    if gene_name:
        gene_isoform_count[gene_name] += 1
    else:
        print(f"⚠️ No gene name found in: {record.description}")  # Debugging output

# Print genes with multiple isoforms
print("Genes with multiple isoforms:")
found = False
for gene, count in gene_isoform_count.items():
    if count > 1:
        print(f"{gene}: {count} isoforms")
        found = True

# Save results to a file
with open("multi_isoform_genes.txt", "w") as out_file:
    for gene, count in gene_isoform_count.items():
        if count > 1:
            out_file.write(f"{gene}: {count} isoforms\n")

if found:
    print("✅ Analysis complete! Check 'multi_isoform_genes.txt' for details.")
else:
    print("❌ No genes with multiple isoforms found.")

