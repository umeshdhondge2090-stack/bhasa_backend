FROM python:3.11-slim

# Hugging Face runs with user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy backend and model files
COPY --chown=user Hindi_Mundari_MT5/ ./Hindi_Mundari_MT5/
COPY --chown=user server.py .

ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860}"]

