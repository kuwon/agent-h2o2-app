REGION=ap-northeast-2
ACCOUNT_ID=037129617559
REPO=h2o2/agent-app
TAG=prd-20250916-amd64-arm64
IMAGE_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO"

# 1) ECR 로그인
aws ecr get-login-password --region $REGION \
| docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# 2) buildx 준비(최초 1회만)
docker buildx create --use || true
docker run --privileged --rm tonistiigi/binfmt --install all

# 3) 멀티아키 빌드 & 푸시 (혹은 amd64만 원하면 linux/amd64만 지정)
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t "$IMAGE_URI:$TAG" \
  --push .

# 4) 배포 스크립트로 새 태그 사용 (태그 고정)
./deploy_image_td_rollout.sh --tag "$TAG" --no-retag --pin-by-digest --health-path /v1/health
