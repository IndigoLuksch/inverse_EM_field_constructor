# tf, gpu install for gc
FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-14.py310:latest

#folder inside container to hold code
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model1.py
COPY train.py

# runs this when the container starts
ENTRYPOINT ["python", "task.py"]