# Projeto de Previsão de Diabetes

## Visão Geral do Projeto
Este projeto implementa um sistema de aprendizado de máquina para prever a probabilidade de um paciente ter diabetes com base em várias variáveis de diagnóstico. O objetivo é classificar os pacientes para prever a presença ou ausência de diabetes.

## Estrutura do Projeto
O projeto está organizado da seguinte forma:
```
├───.gitignore
├───docker-compose.yml
├───Dockerfile
├───main.py
├───README.md
├───requirements.txt
├───content/
│   └───sample_data/
│       ├───diabetes_novo.csv
│       └───diabetes.csv
├───data/
├───notebooks/
│   ├───main_analysis.ipynb
│   ├───refactored-diabetes-eda.ipynb
│   └───refactored-diabetes-pipeline.ipynb
├───src/
│   ├───data_loader.py
│   ├───evaluator.py
│   ├───models.py
│   └───preprocessor.py
└───venv/
```

- **`main.py`**: Ponto de entrada principal para a aplicação.
- **`Dockerfile` e `docker-compose.yml`**: Arquivos de configuração para containerização da aplicação com Docker.
- **`requirements.txt`**: Lista de todas as dependências Python do projeto.
- **`content/sample_data/`**: Contém os conjuntos de dados utilizados no projeto.
- **`notebooks/`**: Jupyter Notebooks para análise exploratória de dados, desenvolvimento e experimentação.
- **`src/`**: Código-fonte modularizado da aplicação.

## Instruções de Configuração

### 1. Clone o Repositório
```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

### 2. Crie e Ative o Ambiente Virtual
```bash
python -m venv venv
source venv/bin/activate  # No Windows, use `venv\Scripts\activate`
```

### 3. Instale as Dependências
```bash
pip install -r requirements.txt
```

## Como Usar
Atualmente, a lógica principal e a análise estão contidas nos Jupyter Notebooks na pasta `notebooks/`. Para executar a análise, inicie o Jupyter Lab:
```bash
jupyter lab
```
Em seguida, navegue até a pasta `notebooks` e abra um dos notebooks.

## Dados
O conjunto de dados utilizado neste projeto é o "Pima Indians Diabetes Database". Ele está localizado em `content/sample_data/` e contém os seguintes arquivos:
- `diabetes.csv`: O conjunto de dados original.
- `diabetes_novo.csv`: Uma versão possivelmente modificada ou estendida do conjunto de dados.

## Notebooks
- **`main_analysis.ipynb`**: Notebook principal com a análise completa, incluindo exploração de dados, pré-processamento, treinamento do modelo e avaliação.
- **`refactored-diabetes-eda.ipynb`**: Notebook focado na Análise Exploratória de Dados (EDA) refatorada.
- **`refactored-diabetes-pipeline.ipynb`**: Notebook que implementa um pipeline de dados refatorado.
