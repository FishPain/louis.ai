#!/bin/bash

# Start the PostgreSQL container
docker run --name pgvector-container \
    -e POSTGRES_USER=langchain \
    -e POSTGRES_PASSWORD=langchain \
    -e POSTGRES_DB=langchain \
    -p 6024:5432 \
    -d pgvector/pgvector:pg16

echo "Starting PostgreSQL container..."

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
until docker exec pgvector-container pg_isready -U langchain -d langchain > /dev/null 2>&1; do
    sleep 2
done

echo "PostgreSQL is ready!"

# Populate VectorDB
echo "Populating Vector DB..."
python -m src.database

echo "Vector DB is ready..."