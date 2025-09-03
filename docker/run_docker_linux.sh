#!/bin/bash

docker compose --env-file .env build

echo "---------------------"
echo "Running clip-init..."
docker compose --env-file .env --profile init run --rm -T clip-init
echo "clip-init finished."

docker compose --env-file .env up -d
