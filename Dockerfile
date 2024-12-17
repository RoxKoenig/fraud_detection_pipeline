# Base image
FROM python:3.12.3

# Set working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    gunicorn \
    psycopg2-binary

# Copy the project files into the container
COPY . .

# Expose Flask server port
EXPOSE 5000

# Start Flask app with Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]

