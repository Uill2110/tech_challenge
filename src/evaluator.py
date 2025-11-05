import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import shap

class ModelEvaluator:
    """
    Avalia modelos de classificação, gera visualizações e fornece insights de interpretabilidade.
    """
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.evaluation_results = {}

    def evaluate_models(self, models, X_test, y_test):
        """
        Avalia um dicionário de modelos e armazena os resultados.
        """
        for name, model in models.items():
            print(f"\n--- Avaliando: {name} ---")
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, y_proba) if y_proba is not None else 'N/A'
            }
            
            self.evaluation_results[name] = metrics

            print(pd.DataFrame([metrics]))
            print("\nRelatório de Classificação:")
            print(classification_report(y_test, y_pred))

            # Matriz de Confusão
            self.plot_confusion_matrix(y_test, y_pred, name)
        
        return self.evaluation_results

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
        plt.title(f'Matriz de Confusão para {model_name}')
        plt.ylabel('Rótulo Verdadeiro')
        plt.xlabel('Rótulo Previsto')
        plt.show()

    def plot_feature_importance(self, model, model_name):
        """
        Plota a importância das features para modelos que a suportam.
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            title = f"Importância das Features para {model_name}"
        elif hasattr(model, 'coef_'):
            importances = model.coef_[0]
            indices = np.argsort(np.abs(importances))[::-1]
            title = f"Coeficientes das Features para {model_name}"
        else:
            print(f"Importância de features não disponível para {model_name}.")
            return

        plt.figure(figsize=(10, 8))
        plt.title(title)
        sns.barplot(x=importances[indices], y=np.array(self.feature_names)[indices])
        plt.xlabel("Importância Relativa / Valor do Coeficiente")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

    def plot_shap_summary(self, model, X_test_processed, model_name):
        """
        Gera um gráfico de resumo SHAP para explicar a importância geral das features.
        """
        print(f"\n--- Análise SHAP para {model_name} ---")
        try:
            explainer = shap.Explainer(model, X_test_processed)
            shap_values = explainer(X_test_processed)

            print("Gerando gráfico de resumo SHAP...")
            shap.summary_plot(shap_values, X_test_processed, feature_names=self.feature_names, show=False)
            plt.title(f"Resumo SHAP para {model_name}")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Não foi possível gerar o gráfico SHAP para {model_name}: {e}")

if __name__ == '__main__':
    from data_loader import DataLoader
    from preprocessor import DataPreprocessor
    from models import ModelTrainer
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    # 1. Carregar e pré-processar dados
    loader = DataLoader()
    df = loader.load_sample_data('diabetes')
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df, target_column='Outcome')
    preprocessor.create_preprocessing_pipeline(X_train_df=X_train)
    X_train_p, X_val_p, X_test_p = preprocessor.preprocess_data(X_train, X_val, X_test)

    # 2. Treinar modelos
    models_to_train = {
        'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5)
    }
    trainer = ModelTrainer()
    trained_models = trainer.train_models(models_to_train, X_train_p, y_train)

    # 3. Avaliar modelos
    evaluator = ModelEvaluator(feature_names=preprocessor.original_feature_names)
    evaluation_summary = evaluator.evaluate_models(trained_models, X_test_p, y_test)

    print("\n--- Resumo da Avaliação ---")
    print(pd.DataFrame(evaluation_summary).T)

    # 4. Plotar importância das features e SHAP
    for name, model in trained_models.items():
        evaluator.plot_feature_importance(model, name)
        evaluator.plot_shap_summary(model, X_test_p, name)
