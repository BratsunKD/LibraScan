FROM python:3.9-slim

WORKDIR /app

RUN pip install gradio requests

COPY ./app/main_front.py .

EXPOSE 7860

CMD ["python", "main_front.py"]