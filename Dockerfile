FROM tiangolo/uwsgi-nginx-flask:python3.8
LABEL maintainer="Rahimeh Neamatian Monemi <contact@predictim-globe.com>"

COPY requirements.txt /tmp/
COPY ./app /app
WORKDIR /app
RUN pip install -U pip && pip install -r /tmp/requirements.txt

ENV DASH_DEBUG_MODE True

ENV NGINX_WORKER_PROCESSES auto

EXPOSE 8050
CMD ["python", "app.py"]
