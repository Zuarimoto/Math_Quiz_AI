# Use a stable Python runtime as a parent image.
FROM python:3.9-slim

# Set the working directory in the container.
WORKDIR /app

# Copy the requirements file and install the dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Expose the port where the app will run.
EXPOSE 8000

# Run the application using Uvicorn.
# The format is "module:app_object_name".
# Your file is main.py, and your app is quiz_api.
CMD ["uvicorn", "main:quiz_api", "--host", "0.0.0.0", "--port", "8000"]