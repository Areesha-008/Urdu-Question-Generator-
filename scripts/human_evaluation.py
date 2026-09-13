import argparse
import csv
import math
from sklearn.metrics import cohen_kappa_score

parser = argparse.ArgumentParser()
parser.add_argument('ratings', help='TSV with id and A_/B_ fluency, relevance, answerability columns')
args = parser.parse_args()
with open(args.ratings, encoding='utf-8', newline='') as file:
    rows = list(csv.DictReader(file, delimiter='\t'))
if not rows or len({row['id'] for row in rows}) != len(rows):
    raise ValueError('Ratings must contain unique sample IDs and at least one row.')
for criterion in ['fluency', 'relevance', 'answerability']:
    ratings = [[int(row[f'{rater}_{criterion}']) for row in rows] for rater in ['A', 'B']]
    if any(value not in (0, 1) for values in ratings for value in values):
        raise ValueError('Use only 0 or 1 ratings.')
    kappa = cohen_kappa_score(*ratings) if len(set(ratings[0] + ratings[1])) > 1 else float('nan')
    print(criterion, 'yes percentages:', [100 * sum(r)/len(r) for r in ratings],
          'kappa:', kappa if math.isfinite(kappa) else 'undefined (no variation)')
