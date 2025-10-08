FROM python:3.11.0-bullseye

ARG GITHUB_TOKEN


# Fix timezone container
ENV TZ=America/Sao_Paulo
ENV TERM=xterm
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


# Add SQL Server ODBC Driver 17 for Debian
# RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
#     RUN curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list
#     RUN apt-get update
#     RUN ACCEPT_EULA=Y apt-get install -y msodbcsql17
#     RUN ACCEPT_EULA=Y apt-get install -y mssql-tools
#     RUN echo 'export PATH="$PATH:/opt/mssql-tools18/bin"' >> ~/.bashrc
    
#     # Install unixodbc for PYODBC
#     RUN apt-get update && apt-get install -y --no-install-recommends \
#         unixodbc-dev \
#         unixodbc \
#         libpq-dev 

# UPDATE APT-GET
RUN apt-get update

RUN apt-get install nano -y

WORKDIR /app

COPY . .

# Substitua "github.com" por "${GITHUB_TOKEN}@github.com" no arquivo
RUN sed -i "s|github.com|${GITHUB_TOKEN}@github.com|g" /app/requirements.txt

RUN pip install -r requirements.txt
