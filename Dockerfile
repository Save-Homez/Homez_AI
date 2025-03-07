# Use an official Python runtime as a base image
FROM python:3.9.17-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=src/app.py

# Run app.py when the container launches
CMD ["gunicorn", "--chdir", "src", "--bind", "0.0.0.0:5000", "app:app"]
