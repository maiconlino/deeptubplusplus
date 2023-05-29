FROM python:3.9-slim

ENV PYTHONUNBUFFERED True

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./
 
RUN pip install Flask gunicorn click lime pandas numpy scipy scikit-learn sqlalchemy cloud-sql-python-connector[pymysql] pymysql bcrypt jsonify
#RUN pip install --upgrade mysql-connector-python


CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app