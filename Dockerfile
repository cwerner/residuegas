FROM python:3.8-slim-buster
COPY app.py /app/
COPY requirements.txt /app/ 
COPY model*v3.pkl.z /app/
#COPY *csv /app/
#COPY permutation_importance.png /app/

WORKDIR /app

RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]

