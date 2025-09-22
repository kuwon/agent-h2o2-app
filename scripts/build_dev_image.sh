#!/bin/bash

set -e

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(dirname ${CURR_DIR})"
DOCKER_FILE="Dockerfile"
REPO="local"
NAME="agent-app"
TAG="dev"

# Run docker buildx create --use before running this script
#echo "Running: docker buildx build --platform=linux/amd64,linux/arm64 -t $REPO/$NAME:$TAG -f $DOCKER_FILE $WS_ROOT --push"
#docker buildx build --platform=linux/amd64,linux/arm64 -t $REPO/$NAME:$TAG -f $DOCKER_FILE $WS_ROOT

#docker buildx ls
# => default * docker
#    h2o2builder    docker-container  (있을 수도, 없을 수도)
# docker buildx ls | grep "\*" | awk -F '\*' '{print $1}'
# 기본(엔진) 빌더로 전환
#docker buildx use default

# BuildKit 끄고(엔진이 직접 빌드), arm64로 로컬 실행
DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 DOCKER_DEFAULT_PLATFORM=linux/arm64 \
ag ws down -y && ag ws up dev -y
