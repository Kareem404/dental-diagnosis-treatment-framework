FROM python:3.12-slim

RUN useradd -m -u 1000 user
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app

USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# run using uvicorn (huggingface exposes port 7860 by default)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"] 