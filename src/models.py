from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

class ModelTrainer:
    """
    Trains various classification models.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}

    def train_logistic_regression(self, X_train, y_train):
        """
        Trains a Logistic Regression model.
        """
        print("Training Logistic Regression model...")
        model = LogisticRegression(random_state=self.random_state, solver='liblinear')
        model.fit(X_train, y_train)
        self.models['LogisticRegression'] = model
        print("Logistic Regression trained.")
        return model

    def train_decision_tree(self, X_train, y_train):
        """
        Trains a Decision Tree Classifier model.
        """
        print("Training Decision Tree model...")
        model = DecisionTreeClassifier(random_state=self.random_state, max_depth=5) # Limiting depth for interpretability
        model.fit(X_train, y_train)
        self.models['DecisionTree'] = model
        print("Decision Tree trained.")
        return model

    def get_models(self):
        """
        Returns all trained models.
        """
        return self.models

if __name__ == "__main__":
    # This block assumes processed data is available
    from data_loader import DataLoader
    from preprocessor import DataPreprocessor

    loader = DataLoader()
    df = loader.load_breast_cancer_data()
    preprocessor = DataPreprocessor()
    X_train, _, _, y_train, _, _ = preprocessor.split_data(df, target_column='target')
    preprocessor.create_preprocessing_pipeline(X_train)
    X_train_p, _, _ = preprocessor.preprocess(X_train, X_train.head(1), X_train.head(1)) # Only need X_train_p here for example

    trainer = ModelTrainer()
    lr_model = trainer.train_logistic_regression(X_train_p, y_train)
    dt_model = trainer.train_decision_tree(X_train_p, y_train)

    print("\nTrained models:", trainer.get_models())