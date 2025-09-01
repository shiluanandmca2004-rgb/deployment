FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and model
COPY . .

EXPOSE 5000

# Start API with gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
