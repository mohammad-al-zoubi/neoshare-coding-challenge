FROM python:3.10

WORKDIR /app

# Copy all files to the working directory
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# Command to run the server
CMD ["python", "-m", "server"]
