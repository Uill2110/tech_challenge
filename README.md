# Tech Challenge — Fase 1

API de ML (FastAPI) para executar um pipeline completo de dados tabulares (carregar, pré-processar, treinar, avaliar e visualizar), com endpoints para uso direto e um relatório HTML único semelhante ao notebook.

Principais recursos:
- Subida via Docker (Dockerfile + docker-compose).
- Documentação Swagger automática em `/docs`.
- Rotas “sem parâmetro” que executam o fluxo padrão e retornam resultados imediatamente.
- Relatório consolidado em `/report` (tabelas + gráficos embutidos).


## Estrutura do projeto

```
docker-compose.yml
Dockerfile
main.py
README.md
requirements.txt
content/
	sample_data/
		diabetes.csv
		diabetes_novo.csv
notebooks/
	main_analysis.ipynb
	refactored-diabetes-eda.ipynb
	refactored-diabetes-pipeline.ipynb
	testes-diabetes_*.ipynb
src/
	data_loader.py
	evaluator.py
	models.py
	preprocessor.py
docs/
	DOCKER.md
```


## Pré-requisitos

- Windows com Docker Desktop instalado e em execução
- Porta `8000` livre no host


## Subir com Docker (Windows PowerShell)

```powershell
# 1) Build da imagem (opcional; o próximo comando também faz build quando necessário)
docker compose build

# 2) Subir o serviço em segundo plano
docker compose up -d --build

# 3) Acompanhar os logs (opcional)
docker compose logs -f

# 4) Testar o healthcheck
curl.exe http://localhost:8000/health

# 5) Parar/derrubar o serviço quando quiser
docker compose down
```

Após subir, acesse:
- Documentação (Swagger): http://localhost:8000/docs
- Relatório HTML completo: http://localhost:8000/report


## Endpoints úteis

- GET `/` — mensagem de boas-vindas
- GET `/health` — healthcheck do container
- GET `/docs` — Swagger UI (testes interativos)
- GET `/report` — executa o pipeline e renderiza página com tabelas e gráficos
- GET `/pipeline/default` — executa carregar + treinar + avaliar e retorna métricas JSON
- GET `/evaluate` — retorna as métricas mais recentes em JSON (após treino)
- Visualizações (PNG):
	- GET `/visualizations/confusion-matrix`
	- GET `/visualizations/feature-importance`
	- GET `/visualizations/shap-summary`

Observações:
- Algumas rotas de visualização exigem que o modelo já tenha sido treinado na sessão atual; usar `/pipeline/default` antes garante o estado necessário.


## Como testar rapidamente

1) Abra o Swagger: http://localhost:8000/docs
2) Rode o endpoint `GET /pipeline/default` para treinar/avaliar com os dados padrão.
3) Acesse o relatório em http://localhost:8000/report para ver tabelas e gráficos.

Opcional (linha de comando):
```powershell
# Executa o pipeline padrão
curl.exe http://localhost:8000/pipeline/default

# Baixa uma imagem de visualização
curl.exe -o conf_matrix.png http://localhost:8000/visualizations/confusion-matrix
```


## Solução de problemas (FAQ)

- Container reiniciando em loop
	- Veja os logs: `docker compose logs -f`
	- Verifique se a porta 8000 está livre (feche outros serviços que possam estar usando-a).
	- Se você alterou arquivos, rode `docker compose up -d --build` para rebuildar a imagem.

- `curl` no PowerShell retorna um objeto em vez do corpo
	- Use `curl.exe` (como acima) ou `Invoke-WebRequest`/`iwr` explicitamente.

- Falha ao instalar dependências durante o build
	- Certifique-se de estar online e que o `requirements.txt` está íntegro.
	- Tente limpar o cache do build adicionando `--no-cache` ao `docker compose build`.


## Desenvolvimento local (opcional)

Se preferir rodar localmente fora do Docker, instale as dependências e execute o Uvicorn:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Então acesse http://localhost:8000/docs ou http://localhost:8000/report.


## Documentação adicional

- Guia de Docker detalhado: `docs/DOCKER.md`
- Notebooks com EDA e pipeline: pasta `notebooks/`


## Licença

Projeto acadêmico/educacional. Ajuste a licença conforme necessidade do seu curso/time.