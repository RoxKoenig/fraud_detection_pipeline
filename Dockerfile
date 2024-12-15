# Use the official Python image
FROM python:3.12.3

# Set the working directory
WORKDIR /app

# Copy the project files into the container
COPY . .

# Expose the port your application runs on
EXPOSE 5000

# Run the application with Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]

