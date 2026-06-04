#!/bin/bash
# Git credential helper that reads GITHUB_TOKEN from .env

ENV_FILE="$(git rev-parse --show-toplevel)/.env"

if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

echo "username=Yogesh-001"
echo "password=${GITHUB_TOKEN}"
