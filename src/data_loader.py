import pandas as pd

class DataLoader:
    """
    Carrega os dados médicos para análise exploratória.
    """
    def __init__(self):
        self.data = None
        self.target_name = None
        self.feature_names = None

    def load_dataset(self, filepath):
        """
        Carrega informacoes do dataset passado como parametro.
        """
        # Importando a base de dados
        dataset = pd.read_csv(filepath)
        self.data = dataset
        self.target_name = 'Outcome'
        self.feature_names = dataset.columns[:-1]
        print(f"Dataset carregado. Shape: {self.data.shape}")
        print(f"Variável target: '{self.target_name}' (1) e 'Negativo para diabetes' (0)")
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
            print("Dataset não carregado.")

if __name__ == "__main__":
    filepath = r"F:\Projetos_Pessoais\fase1\content\sample_data\diabetes.csv"
    loader = DataLoader()
    df = loader.load_dataset(filepath)
    loader.get_info()