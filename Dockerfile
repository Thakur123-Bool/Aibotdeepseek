# Use the official Python Alpine image (smaller base image)
FROM python:3.9-alpine

# Set the working directory in the container
WORKDIR /app

# Install build dependencies and clean up cache to keep the image small
RUN apk add --no-cache --virtual .build-deps gcc musl-dev libffi-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && apk del .build-deps

# Copy the application files into the container
COPY . /app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "App:app", "--host", "0.0.0.0", "--port", "8000"]
