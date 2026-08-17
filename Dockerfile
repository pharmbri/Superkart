# Use a Python 3.10 slim base image for compatibility with numpy==2.0.2
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask application, the serialized model, preprocessor, and feature names
COPY app.py .
COPY tuned_random_forest_model.joblib .
COPY preprocessor.joblib .
COPY all_feature_names.joblib .

# Expose the port Flask runs on
EXPOSE 5000

# Command to run the Flask application
# For production, consider using Gunicorn or uWSGI
CMD ["python", "app.py"]
