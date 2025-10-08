import pandas as pd
from sklearn.datasets import load_breast_cancer

class DataLoader:
    """
    Handles loading of medical datasets and provides initial inspection.
    """
    def __init__(self):
        self.data = None
        self.target_name = None
        self.feature_names = None

    def load_breast_cancer_data(self):
        """
        Loads the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn.
        """
        data = load_breast_cancer(as_frame=True)
        self.data = data.frame
        self.target_name = data.target_names[1] # 'malignant' is 1, 'benign' is 0
        self.feature_names = data.feature_names
        print(f"Dataset loaded. Shape: {self.data.shape}")
        print(f"Target variable: '{data.target_names[0]}' (0) and '{data.target_names[1]}' (1)")
        return self.data

    def get_info(self):
        """
        Prints basic information about the loaded dataset.
        """
        if self.data is not None:
            print("\nDataset Info:")
            self.data.info()
            print("\nFirst 5 rows:")
            print(self.data.head())
            print("\nDescriptive Statistics:")
            print(self.data.describe())
        else:
            print("No data loaded yet.")

if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_breast_cancer_data()
    loader.get_info()