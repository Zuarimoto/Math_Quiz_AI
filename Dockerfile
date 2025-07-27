FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the working directory
COPY . .

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the application using Gunicorn and Uvicorn workers
# Use the wsgi.py entry point and bind to port 80
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "wsgi:app", "-b", "0.0.0.0:80"]
