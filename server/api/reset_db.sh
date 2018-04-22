#!/bin/bash

rm -rf ./shield/v1/alpha/migrations/*
echo "" > ./shield/v1/alpha/migrations/__init__.py
docker container stop ps01
docker container rm ps01
docker-compose run web python manage.py makemigrations
docker-compose run web python manage.py migrate
docker-compose run web python manage.py loaddata fixtures/dev.json
