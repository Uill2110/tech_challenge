FROM python:3.11-bullseye

# Variáveis de ambiente para melhor comportamento no container
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=America/Sao_Paulo \
    TERM=xterm

# Ajuste de timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# Instalar dependências do sistema (usadas por numpy/scipy/scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar apenas requirements primeiro para aproveitar cache
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copiar o restante do código
COPY . .

# Expor a porta do servidor
EXPOSE 8000

# Comando padrão: subir API FastAPI com Uvicorn
CMD ["python", "-c", "import os,sys; import uvicorn; sys.path.append('/app'); uvicorn.run('main:app', host='0.0.0.0', port=int(os.getenv('PORT', '8000')), reload=False)"]
