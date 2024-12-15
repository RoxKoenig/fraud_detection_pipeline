# Use the official Python slim image
FROM python:3.12.3-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY . .

# Expose the port your application runs on
EXPOSE 5000

# Add health check for the application
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl --fail http://localhost:5000/health || exit 1

# Run the application with Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]

