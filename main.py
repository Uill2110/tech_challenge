from typing import Dict, Any, Optional, List, Tuple

import os
import sys
import io
import base64
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

# Garante que o diretório 'src' esteja no PYTHONPATH quando rodar via Uvicorn
SRC_PATH = os.path.join(os.getcwd(), 'src')
if SRC_PATH not in sys.path:
	sys.path.append(SRC_PATH)

# Importa módulos do projeto (src/)
from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.models import ModelTrainer

# Modelos do scikit-learn para exemplo
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Renderização headless de gráficos
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap


# App FastAPI com Swagger habilitado (padrão em /docs)
app = FastAPI(title="Tech Challenge API", version="1.0.0")


# ---------------------------
# Estrutura de estado em memória
# ---------------------------
class AppState:
	"""Guarda objetos e dados necessários entre chamadas (apenas memória)."""
	def __init__(self):
		self.df: Optional[pd.DataFrame] = None
		self.target_column: Optional[str] = None
		self.feature_names: Optional[List[str]] = None

		self.preprocessor: Optional[DataPreprocessor] = None
		self.X_test_processed = None
		self.X_test_raw: Optional[pd.DataFrame] = None
		self.y_test = None

		self.trainer: Optional[ModelTrainer] = None
		self.models: Dict[str, Any] = {}


STATE = AppState()


# ---------------------------
# Esquemas (Pydantic) das requisições
# ---------------------------
class LoadCsvRequest(BaseModel):
	filepath: str = Field(..., description="Caminho do CSV (relativo ao projeto), ex: content/sample_data/diabetes.csv")
	target_column: str = Field(..., description="Nome da coluna alvo do dataset")


class LoadSampleRequest(BaseModel):
	dataset_name: str = Field(..., description="'breast_cancer' ou 'diabetes' (do scikit-learn)")


class TrainRequest(BaseModel):
	test_size: float = Field(0.2, ge=0.05, le=0.4, description="Proporção para teste")
	val_size: float = Field(0.2, ge=0.05, le=0.4, description="Proporção para validação")


class PredictRequest(BaseModel):
	model_name: str = Field(..., description="Nome do modelo treinado")
	features: Dict[str, Any] = Field(..., description="Dicionário de features no formato {nome: valor}")


# ---------------------------
# Rotas básicas
# ---------------------------
@app.get("/")
def read_root():
	"""
	Endpoint raiz: ajuda rápida e links úteis.
	"""
	return {
		"message": "API do Tech Challenge no ar",
		"docs_url": "/docs",
		"openapi_url": "/openapi.json",
	}


@app.get("/health")
def health_check():
	"""Healthcheck simples para o Docker Compose."""
	return {"status": "ok"}


# ---------------------------
# Rotas de dados
# ---------------------------
@app.get("/datasets/sample-files")
def list_sample_csvs():
	"""Lista arquivos CSV em content/sample_data para conveniência."""
	base = os.path.join("content", "sample_data")
	if not os.path.isdir(base):
		return {"files": []}
	files = [f for f in os.listdir(base) if f.lower().endswith(".csv")]
	return {"path": base, "files": files}


@app.post("/load/csv")
def load_from_csv(req: LoadCsvRequest):
	"""
	Carrega dataset de um CSV e guarda no estado.
	"""
	loader = DataLoader()
	csv_path = req.filepath
	if not os.path.isabs(csv_path):
		csv_path = os.path.join(os.getcwd(), csv_path)

	df = loader.load_from_csv(csv_path, target_column=req.target_column)
	if df is None:
		raise HTTPException(status_code=404, detail="Arquivo CSV não encontrado")

	STATE.df = df
	STATE.target_column = req.target_column
	STATE.feature_names = [c for c in df.columns if c != req.target_column]
	# Limpa estado anterior de treino
	STATE.preprocessor = None
	STATE.trainer = None
	STATE.models = {}
	STATE.X_test_processed = None
	STATE.X_test_raw = None
	STATE.y_test = None

	return {"shape": list(df.shape), "target_column": req.target_column, "features": STATE.feature_names}


@app.post("/load/sample")
def load_sample(req: LoadSampleRequest):
	"""
	Carrega dataset de exemplo do scikit-learn ('breast_cancer' ou 'diabetes').
	"""
	loader = DataLoader()
	try:
		df = loader.load_sample_data(req.dataset_name)
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))

	STATE.df = df
	STATE.target_column = "target"
	STATE.feature_names = [c for c in df.columns if c != "target"]
	# Limpa estado anterior de treino
	STATE.preprocessor = None
	STATE.trainer = None
	STATE.models = {}
	STATE.X_test_processed = None
	STATE.X_test_raw = None
	STATE.y_test = None

	return {"dataset": req.dataset_name, "shape": list(df.shape), "target_column": STATE.target_column, "features": STATE.feature_names}


# ---------------------------
# Rotas de treino e avaliação
# ---------------------------
@app.post("/train")
def train_models(req: TrainRequest):
	"""
	Divide dados, cria pipeline, pré-processa e treina três modelos padrão.
	"""
	if STATE.df is None or STATE.target_column is None:
		raise HTTPException(status_code=400, detail="Nenhum dataset carregado. Use /load/csv ou /load/sample primeiro.")

	pre = DataPreprocessor()
	X_train, X_val, X_test, y_train, y_val, y_test = pre.split_data(
		STATE.df, target_column=STATE.target_column, test_size=req.test_size, val_size=req.val_size
	)
	pre.create_preprocessing_pipeline(X_train_df=X_train)
	X_train_p, X_val_p, X_test_p = pre.preprocess_data(X_train, X_val, X_test)

	models_to_train = {
		'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
		'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
		'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
	}

	trainer = ModelTrainer()
	trained = trainer.train_models(models_to_train, X_train_p, y_train)

	# Atualiza estado
	STATE.preprocessor = pre
	STATE.trainer = trainer
	STATE.models = trained
	STATE.X_test_processed = X_test_p
	STATE.X_test_raw = X_test
	STATE.y_test = y_test

	return {"models": list(trained.keys()), "n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test)}


# ---------------------------
# Rotas GET sem parâmetros (defaults)
# ---------------------------
@app.get("/load/sample-default")
def load_sample_default():
	"""Carrega o dataset de exemplo padrão 'breast_cancer'."""
	return load_sample(LoadSampleRequest(dataset_name="breast_cancer"))


@app.get("/load/csv-default")
def load_csv_default():
	"""Carrega o CSV padrão de diabetes com target 'Outcome'."""
	default_path = os.path.join("content", "sample_data", "diabetes.csv")
	return load_from_csv(LoadCsvRequest(filepath=default_path, target_column="Outcome"))


@app.get("/train/default")
def train_default():
	"""Treina modelos com parâmetros padrão (test_size=0.2, val_size=0.2)."""
	return train_models(TrainRequest())


@app.get("/pipeline/default")
def pipeline_default():
	"""
	Executa um pipeline completo padrão: carrega dataset de exemplo, treina modelos e retorna métricas.
	"""
	load_sample_default()
	train_default()
	return evaluate_models()


@app.get("/evaluate")
def evaluate_models():
	"""
	Calcula métricas no conjunto de teste para os modelos treinados.
	"""
	if not STATE.models or STATE.X_test_processed is None or STATE.y_test is None:
		raise HTTPException(status_code=400, detail="Nenhum modelo treinado. Use /train primeiro.")

	results: Dict[str, Dict[str, float]] = {}
	for name, model in STATE.models.items():
		y_pred = model.predict(STATE.X_test_processed)
		proba = None
		if hasattr(model, "predict_proba"):
			try:
				proba = model.predict_proba(STATE.X_test_processed)[:, 1]
			except Exception:
				proba = None
		metrics = {
			"accuracy": float(accuracy_score(STATE.y_test, y_pred)),
			"precision": float(precision_score(STATE.y_test, y_pred)),
			"recall": float(recall_score(STATE.y_test, y_pred)),
			"f1": float(f1_score(STATE.y_test, y_pred)),
			"roc_auc": float(roc_auc_score(STATE.y_test, proba)) if proba is not None else None,
		}
		results[name] = metrics

	return {"metrics": results}


# ---------------------------
# Predição
# ---------------------------
@app.post("/predict")
def predict(req: PredictRequest):
	"""
	Realiza predição para uma única observação usando um modelo treinado.
	Passe as features no formato {nome: valor} com os mesmos nomes usados no dataset.
	"""
	if not STATE.models or STATE.preprocessor is None:
		raise HTTPException(status_code=400, detail="Nenhum modelo treinado. Use /train primeiro.")

	if req.model_name not in STATE.models:
		raise HTTPException(status_code=404, detail=f"Modelo '{req.model_name}' não encontrado.")

	# Garante que todas as features existam (faltantes recebem NaN)
	feature_values = {col: req.features.get(col, None) for col in STATE.feature_names or []}
	df = pd.DataFrame([feature_values])

	# Transforma com o mesmo pipeline
	try:
		X_proc = STATE.preprocessor.preprocessor_pipeline.transform(df)
	except Exception as e:
		raise HTTPException(status_code=400, detail=f"Erro ao transformar dados de entrada: {e}")

	model = STATE.models[req.model_name]
	y_pred = model.predict(X_proc)[0]
	proba = None
	if hasattr(model, "predict_proba"):
		try:
			proba = float(model.predict_proba(X_proc)[0, 1])
		except Exception:
			proba = None

	return {"model": req.model_name, "prediction": int(y_pred), "proba_positive": proba}


@app.get("/predict/example")
def predict_example():
	"""
	Predição de exemplo usando a primeira linha do X_test original e um modelo padrão
	(prioriza 'Random Forest' se disponível).
	"""
	if not STATE.models or STATE.preprocessor is None or STATE.X_test_raw is None:
		raise HTTPException(status_code=400, detail="É necessário treinar antes. Use /train ou /pipeline/default.")

	preferred = "Random Forest"
	model_name = preferred if preferred in STATE.models else next(iter(STATE.models.keys()))

	# Constrói o dicionário de features a partir da primeira linha do X_test bruto
	row = STATE.X_test_raw.iloc[0]
	features = {col: (float(row[col]) if pd.api.types.is_numeric_dtype(type(row[col])) else row[col]) for col in row.index}

	return predict(PredictRequest(model_name=model_name, features=features))


# ---------------------------
# Visualizações (imagens)
# ---------------------------
def _select_model(model_name: Optional[str] = None) -> Tuple[str, Any]:
	if not STATE.models:
		raise HTTPException(status_code=400, detail="Nenhum modelo treinado. Use /train primeiro.")
	if model_name and model_name in STATE.models:
		return model_name, STATE.models[model_name]
	preferred = "Random Forest"
	if preferred in STATE.models:
		return preferred, STATE.models[preferred]
	# fallback: primeiro disponível
	name = next(iter(STATE.models.keys()))
	return name, STATE.models[name]


def _fig_to_png_response() -> StreamingResponse:
	buf = io.BytesIO()
	plt.tight_layout()
	plt.savefig(buf, format='png', bbox_inches='tight')
	plt.close()
	buf.seek(0)
	return StreamingResponse(buf, media_type='image/png')


@app.get("/visualizations/confusion-matrix")
def viz_confusion_matrix(model_name: Optional[str] = Query(None)):
	"""Retorna a matriz de confusão como imagem PNG."""
	if STATE.X_test_processed is None or STATE.y_test is None:
		raise HTTPException(status_code=400, detail="É necessário treinar antes. Use /train primeiro.")
	name, model = _select_model(model_name)
	y_pred = model.predict(STATE.X_test_processed)
	cm = confusion_matrix(STATE.y_test, y_pred)

	plt.figure(figsize=(6, 5))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
	            xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
	plt.title(f'Matriz de Confusão - {name}')
	plt.ylabel('Verdadeiro')
	plt.xlabel('Previsto')
	return _fig_to_png_response()


@app.get("/visualizations/feature-importance")
def viz_feature_importance(model_name: Optional[str] = Query(None)):
	"""Retorna importância de features (ou coeficientes) como imagem PNG."""
	name, model = _select_model(model_name)

	if hasattr(model, 'feature_importances_'):
		importances = model.feature_importances_
		indices = np.argsort(importances)[::-1]
		title = f"Importância das Features - {name}"
		xvals = importances[indices]
		ylabels = np.array(STATE.feature_names or [f"f{i}" for i in range(len(importances))])[indices]
	elif hasattr(model, 'coef_'):
		importances = model.coef_[0]
		indices = np.argsort(np.abs(importances))[::-1]
		title = f"Coeficientes das Features - {name}"
		xvals = importances[indices]
		ylabels = np.array(STATE.feature_names or [f"f{i}" for i in range(len(importances))])[indices]
	else:
		raise HTTPException(status_code=400, detail=f"Importância de features não disponível para {name}.")

	plt.figure(figsize=(10, 8))
	sns.barplot(x=xvals, y=ylabels)
	plt.title(title)
	plt.xlabel("Importância / Coeficiente")
	plt.ylabel("Feature")
	return _fig_to_png_response()


@app.get("/visualizations/shap-summary")
def viz_shap_summary(model_name: Optional[str] = Query(None)):
	"""Retorna gráfico de resumo SHAP como imagem PNG."""
	if STATE.X_test_processed is None:
		raise HTTPException(status_code=400, detail="É necessário treinar antes. Use /train primeiro.")
	name, model = _select_model(model_name)
	try:
		explainer = shap.Explainer(model, STATE.X_test_processed)
		shap_values = explainer(STATE.X_test_processed)
		shap.summary_plot(shap_values, STATE.X_test_processed, feature_names=STATE.feature_names, show=False)
		plt.title(f"Resumo SHAP - {name}")
		return _fig_to_png_response()
	except Exception as e:
		raise HTTPException(status_code=400, detail=f"Não foi possível gerar SHAP: {e}")


def _current_metrics() -> Dict[str, Dict[str, float]]:
	results: Dict[str, Dict[str, float]] = {}
	for name, model in STATE.models.items():
		y_pred = model.predict(STATE.X_test_processed)
		proba = None
		if hasattr(model, "predict_proba"):
			try:
				proba = model.predict_proba(STATE.X_test_processed)[:, 1]
			except Exception:
				proba = None
		results[name] = {
			"accuracy": float(accuracy_score(STATE.y_test, y_pred)),
			"precision": float(precision_score(STATE.y_test, y_pred)),
			"recall": float(recall_score(STATE.y_test, y_pred)),
			"f1": float(f1_score(STATE.y_test, y_pred)),
			"roc_auc": float(roc_auc_score(STATE.y_test, proba)) if proba is not None else None,
		}
	return results


def _fig_to_base64() -> str:
	buf = io.BytesIO()
	plt.tight_layout()
	plt.savefig(buf, format='png', bbox_inches='tight')
	plt.close()
	buf.seek(0)
	return base64.b64encode(buf.read()).decode('ascii')


@app.get("/pipeline/default-outputs")
def pipeline_default_outputs(model_name: Optional[str] = Query(None)):
	"""
	Executa um pipeline padrão e retorna uma lista de "saídas" (text/json/image) no estilo de células do Jupyter.
	"""
	outputs: List[Dict[str, Any]] = []

	# 1) Load sample
	info = load_sample_default()
	outputs.append({"type": "text", "content": f"Carregado dataset {info['dataset']} com shape {info['shape']}"})

	# 2) Train default
	train_info = train_default()
	outputs.append({"type": "text", "content": f"Treinados modelos: {', '.join(train_info['models'])}"})

	# 3) Metrics json
	metrics = _current_metrics()
	outputs.append({"type": "json", "content": metrics})

	# 4) Confusion matrix image
	name, model = _select_model(model_name)
	y_pred = model.predict(STATE.X_test_processed)
	cm = confusion_matrix(STATE.y_test, y_pred)
	plt.figure(figsize=(6, 5))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
	            xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
	plt.title(f'Matriz de Confusão - {name}')
	plt.ylabel('Verdadeiro')
	plt.xlabel('Previsto')
	outputs.append({"type": "image", "title": f"Confusion Matrix - {name}", "mime_type": "image/png", "data_base64": _fig_to_base64()})

	# 5) Feature importance image (se disponível)
	try:
		if hasattr(model, 'feature_importances_'):
			importances = model.feature_importances_
			indices = np.argsort(importances)[::-1]
			title = f"Importância das Features - {name}"
			xvals = importances[indices]
			ylabels = np.array(STATE.feature_names or [f"f{i}" for i in range(len(importances))])[indices]
		elif hasattr(model, 'coef_'):
			importances = model.coef_[0]
			indices = np.argsort(np.abs(importances))[::-1]
			title = f"Coeficientes das Features - {name}"
			xvals = importances[indices]
			ylabels = np.array(STATE.feature_names or [f"f{i}" for i in range(len(importances))])[indices]
		else:
			title = None
			xvals = None
		if xvals is not None:
			plt.figure(figsize=(10, 8))
			sns.barplot(x=xvals, y=ylabels)
			plt.title(title)
			plt.xlabel("Importância / Coeficiente")
			plt.ylabel("Feature")
			outputs.append({"type": "image", "title": title, "mime_type": "image/png", "data_base64": _fig_to_base64()})
	except Exception as e:
		outputs.append({"type": "text", "content": f"Falha ao gerar importância de features: {e}"})

	# 6) SHAP summary image (se possível)
	try:
		explainer = shap.Explainer(model, STATE.X_test_processed)
		shap_values = explainer(STATE.X_test_processed)
		shap.summary_plot(shap_values, STATE.X_test_processed, feature_names=STATE.feature_names, show=False)
		plt.title(f"Resumo SHAP - {name}")
		outputs.append({"type": "image", "title": f"SHAP Summary - {name}", "mime_type": "image/png", "data_base64": _fig_to_base64()})
	except Exception as e:
		outputs.append({"type": "text", "content": f"Não foi possível gerar SHAP: {e}"})

	return JSONResponse(content={"outputs": outputs})


# ---------------------------
# Relatório completo em HTML (estilo Jupyter)
# ---------------------------
def _df_to_html(df: pd.DataFrame, max_rows: int = 10) -> str:
		try:
				return df.head(max_rows).to_html(classes="table table-sm table-striped", border=0)
		except Exception:
				return "<em>Não foi possível renderizar a tabela.</em>"


def _metrics_to_html(metrics: Dict[str, Dict[str, float]]) -> str:
		if not metrics:
				return "<p><em>Sem métricas para exibir.</em></p>"
		# Monta tabela simples
		headers = ["Modelo", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
		rows = []
		for name, m in metrics.items():
			roc_val = m.get("roc_auc")
			roc_cell = "" if roc_val is None else f"{roc_val:.4f}"
			rows.append(
				f"<tr><td>{name}</td><td>{m.get('accuracy'):.4f}</td><td>{m.get('precision'):.4f}</td>"
				f"<td>{m.get('recall'):.4f}</td><td>{m.get('f1'):.4f}</td><td>{roc_cell}</td></tr>"
			)
		return """
		<table class="table table-sm table-bordered">
			<thead><tr>{}</tr></thead>
			<tbody>{}</tbody>
		</table>
		""".format("".join([f"<th>{h}</th>" for h in headers]), "".join(rows))


def _plot_corr_heatmap(df: pd.DataFrame) -> str:
		plt.figure(figsize=(10, 8))
		try:
				corr = df.select_dtypes(include=['number']).corr()
				sns.heatmap(corr, cmap='coolwarm', center=0)
				plt.title('Matriz de Correlação (numéricas)')
		except Exception:
				plt.text(0.5, 0.5, 'Não foi possível gerar a matriz de correlação', ha='center')
		return _fig_to_base64()


@app.get("/report", response_class=HTMLResponse)
def report():
		"""
		Executa o pipeline padrão (sample 'breast_cancer' -> treino -> métricas) e retorna uma página HTML com
		tabelas e gráficos incorporados (base64), simulando a experiência de execução sequencial do Jupyter.
		"""
		# 1) Carregar dataset padrão
		info = load_sample_default()
		# 2) Treinar com defaults
		_ = train_default()
		# 3) Métricas atuais
		metrics = _current_metrics()

		# 4) Gráficos
		# 4.1 Matriz de confusão (modelo preferido)
		name, model = _select_model(None)
		y_pred = model.predict(STATE.X_test_processed)
		cm = confusion_matrix(STATE.y_test, y_pred)
		plt.figure(figsize=(6, 5))
		sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
								xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
		plt.title(f'Matriz de Confusão - {name}')
		plt.ylabel('Verdadeiro')
		plt.xlabel('Previsto')
		conf_b64 = _fig_to_base64()

		# 4.2 Importância de features (se disponível)
		feat_b64 = None
		try:
				if hasattr(model, 'feature_importances_'):
						importances = model.feature_importances_
						indices = np.argsort(importances)[::-1]
						title = f"Importância das Features - {name}"
						xvals = importances[indices]
						ylabels = np.array(STATE.feature_names or [f"f{i}" for i in range(len(importances))])[indices]
				elif hasattr(model, 'coef_'):
						importances = model.coef_[0]
						indices = np.argsort(np.abs(importances))[::-1]
						title = f"Coeficientes das Features - {name}"
						xvals = importances[indices]
						ylabels = np.array(STATE.feature_names or [f"f{i}" for i in range(len(importances))])[indices]
				else:
						xvals = None
				if xvals is not None:
						plt.figure(figsize=(10, 8))
						sns.barplot(x=xvals, y=ylabels)
						plt.title(title)
						plt.xlabel("Importância / Coeficiente")
						plt.ylabel("Feature")
						feat_b64 = _fig_to_base64()
		except Exception:
				feat_b64 = None

		# 4.3 SHAP Summary
		shap_b64 = None
		try:
				explainer = shap.Explainer(model, STATE.X_test_processed)
				shap_values = explainer(STATE.X_test_processed)
				shap.summary_plot(shap_values, STATE.X_test_processed, feature_names=STATE.feature_names, show=False)
				plt.title(f"Resumo SHAP - {name}")
				shap_b64 = _fig_to_base64()
		except Exception:
				shap_b64 = None

		# 4.4 Matriz de correlação (dados originais)
		corr_b64 = _plot_corr_heatmap(STATE.df)

		# 5) Tabelas de dados
		head_html = _df_to_html(STATE.df)
		desc_html = _df_to_html(STATE.df.describe())

		# 6) HTML final (Bootstrap leve via CDN)
		html = f"""
		<!doctype html>
		<html lang='pt-br'>
			<head>
				<meta charset='utf-8'/>
				<meta name='viewport' content='width=device-width, initial-scale=1'/>
				<title>Tech Challenge - Relatório</title>
				<link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css' rel='stylesheet'/>
				<style>
					body {{ padding: 20px; }}
					img {{ max-width: 100%; height: auto; border: 1px solid #ddd; padding: 4px; }}
					.section {{ margin-bottom: 32px; }}
					.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
				</style>
			</head>
			<body>
				<h1>Relatório - Execução Automática</h1>
				<p class='text-muted'>Dataset: <strong>{info['dataset']}</strong> | Shape: {info['shape']}</p>

				<div class='section'>
					<h2>Prévia dos dados</h2>
					{head_html}
				</div>

				<div class='section'>
					<h2>Estatísticas descritivas</h2>
					{desc_html}
				</div>

				<div class='section'>
					<h2>Métricas (Teste)</h2>
					{_metrics_to_html(metrics)}
				</div>

				<div class='section'>
					<h2>Gráficos</h2>
					<div class='grid'>
						<div>
							<h5>Matriz de confusão - {name}</h5>
							<img src='data:image/png;base64,{conf_b64}' alt='Confusion Matrix'/>
						</div>
						<div>
							<h5>Matriz de correlação</h5>
							<img src='data:image/png;base64,{corr_b64}' alt='Correlation Heatmap'/>
						</div>
						{f"<div><h5>Importância das features - {name}</h5><img src='data:image/png;base64,{feat_b64}' alt='Feature Importance'/></div>" if feat_b64 else ''}
						{f"<div><h5>SHAP Summary - {name}</h5><img src='data:image/png;base64,{shap_b64}' alt='SHAP Summary'/></div>" if shap_b64 else ''}
					</div>
				</div>

				<p class='text-muted'>Gerado automaticamente pela API FastAPI. Use <a href='/docs'>/docs</a> para explorar as rotas.</p>
			</body>
		</html>
		"""
		return HTMLResponse(content=html)

