import csv

def transform_column_headers(header_row):
    def transform_entry(entry):
        parts = entry.strip().split(' ')
        transformed = []
        for part in parts:
            if part.count('.') >= 2:
                first_dot = part.find('.')
                second_dot = part.find('.', first_dot + 1)
                part = part[:first_dot] + '(' + part[first_dot+1:second_dot] + ')' + part[second_dot+1:]
            transformed.append(part)
        return ' '.join(transformed)

    return [transform_entry(col) for col in header_row]

# Read and write the CSV
with open('C:/Users/tnaom/OneDrive/Desktop/PPA/04_clustering/results/clustered_matrix.csv', 'r', newline='') as infile, \
     open('C:/Users/tnaom/OneDrive/Desktop/PPA/04_clustering/results/clustered_matrix_r.csv', 'w', newline='') as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for i, row in enumerate(reader):
        if i == 0:
            # Transform the header row
            transformed_header = transform_column_headers(row)
            writer.writerow(transformed_header)
        else:
            writer.writerow(row)
