FROM python:3.9-alpine

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the working directory and copy files
WORKDIR /app
COPY . /app

EXPOSE 80

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app:app"]
