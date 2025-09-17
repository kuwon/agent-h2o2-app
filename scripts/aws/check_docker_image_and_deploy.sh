#!/usr/bin/env bash
set -euo pipefail

REGION="ap-northeast-2"
ACCOUNT_ID="037129617559"
REPO="h2o2/agent-app"
TAG="${1:-prd}"  # 인자로 태그 넘기면 사용

IMAGE_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO"

echo "[1] Ensure repository exists"
aws ecr describe-repositories --region "$REGION" --repository-names "$REPO" >/dev/null 2>&1 \
  || aws ecr create-repository --region "$REGION" --repository-name "$REPO"

echo "[2] ECR login"
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

echo "[3] Build"
docker build -t "$IMAGE_URI:$TAG" .

echo "[4] Push"
docker push "$IMAGE_URI:$TAG"

echo "Done -> $IMAGE_URI:$TAG"
