import csv

def transform_multi_entry_field(field):
    parts = field.strip().split(' ')
    transformed = []
    for part in parts:
        if part.count('.') >= 2:
            first_dot = part.find('.')
            second_dot = part.find('.', first_dot + 1)
            part = part[:first_dot] + '(' + part[first_dot+1:second_dot] + ')' + part[second_dot+1:]
        transformed.append(part)
    return ' '.join(transformed)

# Input and output file paths
input_file = 'C:/Users/tnaom/OneDrive/Desktop/PPA/05_feature_selection/interim_data/top_500_fisher_scores_min200vals.csv'
output_file = 'C:/Users/tnaom/OneDrive/Desktop/PPA/05_feature_selection/interim_data/top_500_fisher_scores_min200vals_t.csv'

with open(input_file, 'r', newline='') as infile, \
     open(output_file, 'w', newline='') as outfile:

    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        row['Feature'] = transform_multi_entry_field(row['Feature'])
        row['TargetFeature'] = transform_multi_entry_field(row['TargetFeature'])
        writer.writerow(row)
