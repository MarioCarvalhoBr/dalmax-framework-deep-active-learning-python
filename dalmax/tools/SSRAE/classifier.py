import pickle
import argparse

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, cross_val_score
from numpy import mean

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--pkl', default=None, type=str, help='Specify the path of the pickle object containing the extracted features')

    return parser.parse_args()

if __name__ == '__main__':
    # Parse the input command-line arguments.
    args = parse_args()

    if args.pkl is None:
        raise ValueError('No path has been specified.')

    if args.pkl == '':
        raise ValueError('The path could not be empty.')

    with open(args.pkl, 'rb') as f:
        # Load the pickle object content.
        data = pickle.load(f)
        
        # Check if the feature matrix is not provided inside the pickle object.
        if not 'X' in data:
            raise ValueError('The key `X` (feature matrix) was expected inside the pickle object.')
        
        # Check if the labels is not provided inside the pickle object.
        if not 'y' in data:
            raise ValueError('The key `y` (labels) was expceted inside the pickle object.')
        
        X = data['X']  # The feature matrix.
        y = data['y']  # The labels.

    # Use Linear Discriminant Analysis (LDA) classifier.
    lda = LinearDiscriminantAnalysis()

    # Use the leave-one-out cross-validation strategy.
    cv = LeaveOneOut()

    # Print classifying information.
    print()
    print('Classifying Information:')
    print(f'├── Classifier:                Linear Discriminant Analysis (LDA)')
    print(f'├── Cross-validation Strategy: Leave-one-out')
    print(f'├── PKL Information:')
    print(f'│   ├── Number of Attributes:  {len(X[0])} attributes')
    print(f'│   └── Number of Samples:     {len(y)} samples')

    # Calculates the score.    
    scores = cross_val_score(lda, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    
    print(f'└── Scores Information:')
    print(f'    └── Accuracy:              {(mean(scores) * 100.0):0.02f}%')
    print()