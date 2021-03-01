FROM public.ecr.aws/lambda/python:3.8

# Copy the earlier created requirements.txt file to the container
COPY requirements.txt ./

# Install the python requirements from requirements.txt
RUN python3.8 -m pip install -r requirements.txt

# Copy the earlier created app.py file to the container
COPY app.py ./
COPY src ./src
COPY models/banknote_best.onnx ./models/banknote_best.onnx

# Set the CMD to your handler
CMD ["app.lambda_handler"]