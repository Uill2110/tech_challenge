import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import shap # For model interpretability

class ModelEvaluator:
    """
    Evaluates trained models and provides interpretability.
    """
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Evaluates a single model and prints key metrics.
        """
        y_pred = model.predict(X_test)

        print(f"\n--- Evaluation for {model_name} ---")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Benign (0)', 'Malignant (1)'],
                    yticklabels=['Benign (0)', 'Malignant (1)'])
        plt.title(f'Confusion Matrix for {model_name}')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.show()

        # Critical Discussion on Metrics:
        print("\n--- Critical Discussion on Metrics ---")
        print(f"For breast cancer diagnosis, 'Recall' is often a critical metric.")
        print(f"A high recall ensures that we minimize 'False Negatives' (missing actual cancer cases),")
        print(f"which is crucial even if it means a slightly higher rate of 'False Positives' (benign cases flagged as malignant),")
        print(f"as false positives can be clarified with further tests, whereas false negatives can be life-threatening.")


    def plot_feature_importance(self, model, model_name="Model"):
        """
        Plots feature importance for models that support it (e.g., Decision Trees, Logistic Regression coefficients).
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 6))
            plt.title(f"Feature Importance for {model_name}")
            sns.barplot(x=importances[indices], y=[self.feature_names[i] for i in indices])
            plt.xlabel("Relative Importance")
            plt.ylabel("Feature Name")
            plt.tight_layout()
            plt.show()
        elif hasattr(model, 'coef_'):
            # For Logistic Regression, coefficients can indicate importance
            coef = model.coef_[0] # Assuming binary classification
            feature_coef = pd.Series(coef, index=self.feature_names)
            feature_coef = feature_coef.reindex(feature_coef.abs().sort_values(ascending=False).index)

            plt.figure(figsize=(10, 6))
            feature_coef.plot(kind='barh', color=(feature_coef > 0).map({True: 'skyblue', False: 'salmon'}))
            plt.title(f"Feature Coefficients (Importance) for {model_name}")
            plt.xlabel("Coefficient Value (Magnitude indicates importance, sign indicates direction)")
            plt.ylabel("Feature Name")
            plt.tight_layout()
            plt.show()
        else:
            print(f"Feature importance not directly available for {model_name}.")

    def plot_shap_values(self, model, X_test_processed, original_X_test, model_name="Model"):
        """
        Uses SHAP to explain individual predictions and overall feature importance.
        """
        print(f"\n--- SHAP Explanations for {model_name} ---")
        try:
            # SHAP Explainer
            explainer = shap.Explainer(model, X_test_processed)
            shap_values = explainer(X_test_processed)

            # Summary plot (overall feature importance)
            print("Generating SHAP summary plot...")
            shap.summary_plot(shap_values, X_test_processed, feature_names=self.feature_names, show=False)
            plt.title(f"SHAP Summary Plot for {model_name}")
            plt.tight_layout()
            plt.show()

            # Example individual prediction explanation
            print("Generating SHAP force plot for an example prediction...")
            sample_idx = 0 # Explain the first sample in the test set
            print(f"Explaining prediction for sample {sample_idx}. Actual: {original_X_test.iloc[sample_idx]['target']},"
                  f" Predicted: {model.predict(X_test_processed[sample_idx].reshape(1, -1))[0]}")
            shap.force_plot(explainer.expected_value, shap_values[sample_idx], X_test_processed[sample_idx],
                            feature_names=self.feature_names, show=False, matplotlib=True)
            plt.title(f"SHAP Force Plot for Sample {sample_idx} ({model_name})")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Could not generate SHAP plots for {model_name}: {e}")
            print("Please ensure your model and data are compatible with SHAP.")

if __name__ == "__main__":
    from data_loader import DataLoader
    from preprocessor import DataPreprocessor
    from models import ModelTrainer
    from sklearn.datasets import load_breast_cancer

    # Load data and get original feature names
    data = load_breast_cancer(as_frame=True)
    original_feature_names = data.feature_names
    df = data.frame

    # Preprocess data
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df, target_column='target')
    preprocessor_pipeline = preprocessor.create_preprocessing_pipeline(X_train)
    X_train_p, X_val_p, X_test_p = preprocessor.preprocess(X_train, X_val, X_test)

    # Train models
    trainer = ModelTrainer()
    lr_model = trainer.train_logistic_regression(X_train_p, y_train)
    dt_model = trainer.train_decision_tree(X_train_p, y_train)

    # Combine X_test and y_test for easier SHAP access to original values
    original_X_test_with_target = X_test.copy()
    original_X_test_with_target['target'] = y_test

    # Evaluate models
    evaluator = ModelEvaluator(feature_names=original_feature_names)

    # Logistic Regression Evaluation
    evaluator.evaluate_model(lr_model, X_test_p, y_test, model_name="Logistic Regression")
    evaluator.plot_feature_importance(lr_model, model_name="Logistic Regression")
    # evaluator.plot_shap_values(lr_model, X_test_p, original_X_test_with_target, model_name="Logistic Regression") # SHAP for LR can be complex with pipeline steps

    # Decision Tree Evaluation
    evaluator.evaluate_model(dt_model, X_test_p, y_test, model_name="Decision Tree")
    evaluator.plot_feature_importance(dt_model, model_name="Decision Tree")
    evaluator.plot_shap_values(dt_model, X_test_p, original_X_test_with_target, model_name="Decision Tree")