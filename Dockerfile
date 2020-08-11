FROM python:3.8

WORKDIR /app

COPY src/ src
COPY setup.py .
COPY requirements.txt .
COPY app/ app
COPY models/ models
COPY data/ data

RUN python -m pip install --upgrade pip
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

EXPOSE 8080
EXPOSE 8081

CMD python /app/app/model_view.py

CMD python /app/app/app.py