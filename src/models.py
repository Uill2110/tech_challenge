from sklearn.base import is_classifier

class ModelTrainer:
    """
    Treina uma coleção de modelos de classificação a partir de um dicionário.
    """
    def __init__(self):
        self.trained_models = {}

    def train_models(self, models_dict, X_train, y_train):
        """
        Treina uma série de modelos de classificação.

        Args:
            models_dict (dict): Um dicionário onde as chaves são os nomes dos modelos
                              e os valores são os objetos de modelo (ex: LogisticRegression()).
            X_train: Os dados de treino (features).
            y_train: Os rótulos de treino (target).

        Returns:
            dict: Um dicionário contendo os modelos treinados.
        """
        for name, model in models_dict.items():
            if not is_classifier(model):
                print(f"Aviso: O modelo '{name}' pode não ser um classificador do Scikit-learn. Pulando.")
                continue
            
            print(f"Treinando o modelo: {name}...")
            try:
                model.fit(X_train, y_train)
                self.trained_models[name] = model
                print(f"{name} treinado com sucesso.")
            except Exception as e:
                print(f"Falha ao treinar o modelo {name}. Erro: {e}")
        
        return self.trained_models

    def get_trained_models(self):
        """
        Retorna o dicionário de modelos que foram treinados.
        """
        return self.trained_models

if __name__ == '__main__':
    from data_loader import DataLoader
    from preprocessor import DataPreprocessor
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    # 1. Carregar dados
    loader = DataLoader()
    df = loader.load_sample_data('breast_cancer')

    # 2. Pré-processar dados
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df, target_column='target')
    preprocessor.create_preprocessing_pipeline(X_train_df=X_train)
    X_train_p, _, _ = preprocessor.preprocess_data(X_train, X_val, X_test)

    # 3. Definir os modelos a serem treinados
    models_to_train = {
        'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }

    # 4. Treinar os modelos
    trainer = ModelTrainer()
    trained_classifiers = trainer.train_models(models_to_train, X_train_p, y_train)

    print("\nModelos treinados disponíveis:", list(trained_classifiers.keys()))

    # Exemplo de como acessar um modelo treinado
    if 'Logistic Regression' in trained_classifiers:
        lr_model = trained_classifiers['Logistic Regression']
        print("\nCoeficientes da Regressão Logística:", lr_model.coef_)
