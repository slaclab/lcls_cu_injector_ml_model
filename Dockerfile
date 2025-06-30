FROM python:3.11-slim
WORKDIR /app
# Set env variable for k2eg
ENV K2EG_PYTHON_CONFIGURATION_PATH_FOLDER=/app/config
# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
# Copy project files
COPY . .
# Set working directory to the folder where code lives
WORKDIR /app/k2eg
# Default command to run script
CMD ["python", "k2eg_output.py"]
