import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    """
    Realiza a limpeza, transformação e divisão dos dados, com pipelines configuráveis
    para features numéricas e categóricas.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.preprocessor_pipeline = None
        self.original_feature_names = None

    def handle_zero_as_nan(self, df, columns):
        """
        Substitui valores '0' por NaN em colunas específicas.

        Args:
            df (pd.DataFrame): O DataFrame a ser modificado.
            columns (list): A lista de nomes de colunas a serem processadas.

        Returns:
            pd.DataFrame: O DataFrame com os valores '0' substituídos por NaN.
        """
        df_copy = df.copy()
        for col in columns:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].replace(0, np.nan)
        print(f"Valores '0' substituídos por NaN nas colunas: {columns}")
        return df_copy

    def split_data(self, df, target_column, test_size=0.2, val_size=0.2):
        """
        Divide os dados em conjuntos de treino, validação e teste.

        Args:
            df (pd.DataFrame): O DataFrame completo.
            target_column (str): O nome da coluna de destino.
            test_size (float): A proporção do conjunto de teste.
            val_size (float): A proporção do conjunto de validação (em relação ao total).

        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        self.original_feature_names = X.columns.tolist()

        # Garante que a soma de test_size e val_size não exceda 1
        if test_size + val_size >= 1.0:
            raise ValueError("A soma de test_size e val_size deve ser menor que 1.")

        # Primeira divisão para separar o conjunto de treino
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), stratify=y, random_state=self.random_state
        )

        # Segunda divisão para separar validação e teste
        # Ajusta o tamanho do teste para ser uma proporção do conjunto temporário
        relative_test_size = test_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=relative_test_size, stratify=y_temp, random_state=self.random_state
        )

        print(f"Dados divididos em: Treino ({len(X_train)}), Validação ({len(X_val)}), Teste ({len(X_test)}) amostras.")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_preprocessing_pipeline(self, numerical_features=None, categorical_features=None, X_train_df=None):
        """
        Cria um pipeline de pré-processamento para features numéricas e categóricas.
        Se as listas de features não forem fornecidas, elas são inferidas do DataFrame de treino.
        """
        if numerical_features is None and categorical_features is None and X_train_df is not None:
            numerical_features = X_train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = X_train_df.select_dtypes(include=['object', 'category']).columns.tolist()
            print("Features numéricas e categóricas inferidas automaticamente.")

        # Pipeline para features numéricas
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Preenche valores ausentes com a média
            ('scaler', StandardScaler())                   # Padroniza as features
        ])

        # Pipeline para features categóricas
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')), # Preenche valores ausentes com o mais frequente
            ('onehot', OneHotEncoder(handle_unknown='ignore'))     # Aplica One-Hot Encoding
        ])

        transformers = []
        if numerical_features:
            transformers.append(('num', numeric_transformer, numerical_features))
        if categorical_features:
            transformers.append(('cat', categorical_transformer, categorical_features))

        self.preprocessor_pipeline = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'  # Mantém outras colunas, se houver
        )
        print("Pipeline de pré-processamento criado.")
        return self.preprocessor_pipeline

    def preprocess_data(self, X_train, X_val, X_test):
        """
        Aplica o pipeline de pré-processamento aos conjuntos de dados.
        """
        if self.preprocessor_pipeline is None:
            raise ValueError("Pipeline de pré-processamento não foi criado. Chame create_preprocessing_pipeline primeiro.")

        X_train_processed = self.preprocessor_pipeline.fit_transform(X_train)
        X_val_processed = self.preprocessor_pipeline.transform(X_val)
        X_test_processed = self.preprocessor_pipeline.transform(X_test)

        print("Dados pré-processados com sucesso.")
        return X_train_processed, X_val_processed, X_test_processed

if __name__ == '__main__':
    from data_loader import DataLoader

    # Carregando o dataset de exemplo 'diabetes'
    loader = DataLoader()
    df = loader.load_sample_data('diabetes')

    # Instanciando o pré-processador
    preprocessor = DataPreprocessor()

    # Dividindo os dados
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df, target_column='Outcome')

    # Criando o pipeline (inferindo as colunas a partir do X_train)
    preprocessor.create_preprocessing_pipeline(X_train_df=X_train)

    # Pré-processando os dados
    X_train_p, X_val_p, X_test_p = preprocessor.preprocess_data(X_train, X_val, X_test)

    print("\nShape dos dados de treino processados:", X_train_p.shape)
    print("Shape dos dados de validação processados:", X_val_p.shape)
    print("Shape dos dados de teste processados:", X_test_p.shape)
    print("Nomes das features originais:", preprocessor.original_feature_names)
