
import pandas as pd
from sklearn.datasets import load_diabetes

class DataLoader:
    """
    Classe para carregar dados de várias fontes, como arquivos CSV ou datasets de exemplo do Scikit-learn.
    """
    def __init__(self):
        self.data = None
        self.target_name = None
        self.feature_names = None

    def load_from_csv(self, filepath, target_column):
        """
        Carrega um dataset a partir de um arquivo CSV.

        Args:
            filepath (str): O caminho para o arquivo CSV.
            target_column (str): O nome da coluna de destino (target).
        """
        try:
            dataset = pd.read_csv(filepath)
            self.data = dataset
            self.target_name = target_column
            self.feature_names = [col for col in dataset.columns if col != target_column]
            print(f"Dataset carregado de {filepath}. Shape: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            print(f"Erro: O arquivo não foi encontrado em '{filepath}'")
            return None

    def load_sample_data(self, dataset_name='diabetes'):
        """
        Carrega um dataset de exemplo do Scikit-learn.

        Args:
            dataset_name (str): O nome do dataset ('diabetes').
        """
        if dataset_name == 'diabetes':
            data_loader = load_diabetes
        else:
            raise ValueError("Dataset de exemplo não suportado. Escolha 'diabetes'.")

        sample_data = data_loader()
        self.data = pd.DataFrame(data=sample_data.data, columns=sample_data.feature_names)
        self.data['target'] = sample_data.target
        self.target_name = 'target'
        self.feature_names = sample_data.feature_names
        print(f"Dataset de exemplo '{dataset_name}' carregado. Shape: {self.data.shape}")
        return self.data

    def get_info(self):
        """
        Mostra informações básicas sobre o dataset carregado.
        """
        if self.data is not None:
            print("\nInformações sobre o Dataset:")
            self.data.info()
            print("\nPrimeiras 5 linhas:")
            print(self.data.head())
            print("\nDescrição estatística:")
            print(self.data.describe())
        else:
            print("Nenhum dataset carregado. Use load_from_csv() ou load_sample_data() primeiro.")

if __name__ == '__main__':
    # Exemplo de uso: Carregando do CSV
    print("--- Exemplo: Carregando dados de diabetes de um arquivo CSV ---")
    csv_loader = DataLoader()
    diabetes_csv_path = r"../content/sample_data/diabetes.csv"
    diabetes_df = csv_loader.load_from_csv(diabetes_csv_path, target_column='Outcome')
    if diabetes_df is not None:
        csv_loader.get_info()


