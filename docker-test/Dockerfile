FROM flwr/superexec:1.23.0

WORKDIR /app

COPY . .
RUN python -m pip install -U --no-cache-dir .

ENTRYPOINT ["flower-superexec"]