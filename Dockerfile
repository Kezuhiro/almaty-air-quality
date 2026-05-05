FROM python:3.11-slim


RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app


COPY requirements.txt .

# Устанавливаем строго CPU-версию PyTorch, чтобы сэкономить место
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt


COPY . .


EXPOSE 8000

WORKDIR /app/app


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]