# Execução via Docker

Pré-requisitos:
- Docker Desktop instalado e em execução (Windows)

## Comandos (PowerShell)

```powershell
# 1) Build da imagem
docker compose build

# 2) Subir o serviço
docker compose up -d

# 3) Verificar status do container
docker ps

# 4) Testar healthcheck
curl http://localhost:8000/health

# 5) Logs (opcional)
docker compose logs -f

# 6) Derrubar o serviço (quando quiser parar)
docker compose down
```

## Endpoints iniciais
- GET http://localhost:8000/ → mensagem de boas-vindas
- GET http://localhost:8000/health → healthcheck simples (usado pelo Compose)

## Observações
- O arquivo `requirements.txt` inclui FastAPI e Uvicorn.
- O `Dockerfile` executa a API via Uvicorn na porta 8000.
- O `docker-compose.yml` sobe um único serviço `api` expondo a porta 8000.
- Caso queira montar diretórios de dados/resultados como volumes, podemos adicionar depois conforme necessidade.
