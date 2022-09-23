FROM python:3.8

WORKDIR /home/easy

COPY easy/requirements.txt .

RUN pip install -r requirements.txt

COPY  . .

CMD ["python3", "train.py"]