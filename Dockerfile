FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Bake the local (BM25 + LSA) index into the image so it starts without any API key.
# For an API-embedding index, build it outside and mount it: -v $PWD/artifacts/index-api:/index -e RAG_INDEX_DIR=/index
RUN hybrid-rag download --dest /app/data/peps \
    && hybrid-rag build-index --corpus /app/data/peps --out /app/artifacts/index --embedder lsa:256

ENV RAG_INDEX_DIR=/app/artifacts/index
EXPOSE 8000

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

CMD ["uvicorn", "hybrid_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
