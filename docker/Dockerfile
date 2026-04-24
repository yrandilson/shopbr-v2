FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "import sys; sys.path.insert(0,'.'); \
    from src.data.gerar_dados import gerar; gerar()" && \
    python -c "import sys,os; sys.path.insert(0,'.'); \
    os.makedirs('data/processed',exist_ok=True); \
    from src.features.feature_engineering import criar_features_categoria, criar_features_sku; \
    import pandas as pd; \
    df=criar_features_categoria(); df.to_csv('data/processed/features_categoria.csv',index=False); \
    df2=criar_features_sku(); df2.to_csv('data/processed/features_sku.csv',index=False)" && \
    python -c "import sys; sys.path.insert(0,'.'); \
    from src.models.trainer import treinar; \
    treinar('categoria',forcar_promocao=True)"

EXPOSE 5050
CMD ["python", "dashboard/app.py"]
