import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    """
    Performs data cleaning, transformation, and splitting.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.preprocessor_pipeline = None

    def create_preprocessing_pipeline(self, X):
        """
        Creates a preprocessing pipeline for numerical and categorical features.
        Assumes 'target' is the target column and all others are features.
        """
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
        # For this dataset, all are numerical. If you had categorical, you'd list them here.
        # categorical_features = X.select_dtypes(include=['object', 'category']).columns

        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')), # Handle potential NaNs
            ('scaler', StandardScaler())                  # Scale numerical features
        ])

        # If you had categorical features:
        # categorical_transformer = Pipeline(steps=[
        #    ('imputer', SimpleImputer(strategy='most_frequent')),
        #    ('onehot', OneHotEncoder(handle_unknown='ignore'))
        # ])

        self.preprocessor_pipeline = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                # ('cat', categorical_transformer, categorical_features) # Add if categorical features exist
            ],
            remainder='passthrough' # Keep other columns (if any)
        )
        print("Preprocessing pipeline created.")
        return self.preprocessor_pipeline

    def split_data(self, df, target_column='target', test_size=0.2, val_size=0.2):
        """
        Splits data into training, validation, and test sets.
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # First split: train and (test+validation)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), stratify=y, random_state=self.random_state
        )

        # Second split: test and validation from X_temp
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_size / (test_size + val_size), stratify=y_temp, random_state=self.random_state
        )

        print(f"Data split into: Train ({len(X_train)}), Validation ({len(X_val)}), Test ({len(X_test)}) samples.")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def preprocess(self, X_train, X_val, X_test):
        """
        Applies the preprocessing pipeline to the datasets.
        """
        if self.preprocessor_pipeline is None:
            raise ValueError("Preprocessing pipeline not created. Call create_preprocessing_pipeline first.")

        X_train_processed = self.preprocessor_pipeline.fit_transform(X_train)
        X_val_processed = self.preprocessor_pipeline.transform(X_val)
        X_test_processed = self.preprocessor_pipeline.transform(X_test)

        print("Data preprocessed.")
        return X_train_processed, X_val_processed, X_test_processed

if __name__ == "__main__":
    from data_loader import DataLoader

    loader = DataLoader()
    df = loader.load_breast_cancer_data()

    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df, target_column='target')

    preprocessor.create_preprocessing_pipeline(X_train)
    X_train_p, X_val_p, X_test_p = preprocessor.preprocess(X_train, X_val, X_test)

    print("\nShape of processed training data:", X_train_p.shape)
    print("Shape of processed validation data:", X_val_p.shape)
    print("Shape of processed test data:", X_test_p.shape)