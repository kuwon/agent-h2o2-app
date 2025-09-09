#!/usr/bin/env bash
set -euo pipefail

REGION=${REGION:-ap-northeast-2}
CLUSTER=${CLUSTER:-h2o2-cluster}
SERVICE=${SERVICE:-h2o2-service}
LB_NAME=${LB_NAME:-h2o2-alb}

echo "== 1) 서비스가 실제로 연결한 Target Group 확인"
SVC_TG=$(aws ecs describe-services --region "$REGION" \
  --cluster "$CLUSTER" --services "$SERVICE" \
  --query 'services[0].loadBalancers[0].targetGroupArn' --output text)
echo "Service TargetGroupArn: $SVC_TG"

echo "== 2) ALB 80 리스너가 현재 포워드하는 Target Group 확인"
LB_ARN=$(aws elbv2 describe-load-balancers --region "$REGION" --names "$LB_NAME" \
  --query 'LoadBalancers[0].LoadBalancerArn' --output text)
LISTENER80=$(aws elbv2 describe-listeners --region "$REGION" \
  --load-balancer-arn "$LB_ARN" \
  --query 'Listeners[?Port==`80`].ListenerArn' --output text)
LB_TG=$(aws elbv2 describe-listeners --region "$REGION" --listener-arn "$LISTENER80" \
  --query 'Listeners[0].DefaultActions[0].TargetGroupArn' --output text)
echo "Listener(80) TargetGroupArn: $LB_TG"

if [[ "$SVC_TG" == "None" || -z "$SVC_TG" ]]; then
  echo "⚠️  서비스에 Target Group이 연결되어 있지 않습니다. (create/update-service 때 loadBalancers 인자 필요)"
  exit 1
fi

# 2-1) 리스너가 다른 TG를 보고 있으면 서비스 TG로 맞춤
if [[ "$LB_TG" != "$SVC_TG" ]]; then
  echo "== 3) 리스너 기본 액션을 서비스 TG로 교체"
  aws elbv2 modify-listener --region "$REGION" --listener-arn "$LISTENER80" \
    --default-actions "Type=forward,TargetGroupArn=$SVC_TG" >/dev/null
  echo "✔️  Listener default action -> $SVC_TG 로 일치시킴"
else
  echo "✔️  Listener 기본 대상과 서비스 TG가 이미 일치"
fi

echo "== 4) 헬스체크 조건 완화(/, 200-399) + 드레인 타임아웃 단축"
aws elbv2 modify-target-group --region "$REGION" --target-group-arn "$SVC_TG" \
  --health-check-path "/" --matcher HttpCode=200-399 \
  --health-check-interval-seconds 15 --healthy-threshold-count 2 --unhealthy-threshold-count 2 >/dev/null || true

aws elbv2 modify-target-group-attributes --region "$REGION" --target-group-arn "$SVC_TG" \
  --attributes Key=deregistration_delay.timeout_seconds,Value=30 >/dev/null || true

echo "== 5) 강제 재배포(새 태스크가 TG에 등록되도록)"
aws ecs update-service --region "$REGION" --cluster "$CLUSTER" --service "$SERVICE" --force-new-deployment >/dev/null

echo "== 6) 타깃 헬스 모니터링 (최대 2분)"
for i in {1..24}; do
  STATES=$(aws elbv2 describe-target-health --region "$REGION" --target-group-arn "$SVC_TG" \
    --query 'TargetHealthDescriptions[].TargetHealth.State' --output text || true)
  echo "  [$i] $STATES"
  if echo "$STATES" | grep -q healthy; then
    echo "✔️  Healthy 확인"
    break
  fi
  sleep 5
done

ALB_DNS=$(aws elbv2 describe-load-balancers --region "$REGION" --load-balancer-arns "$LB_ARN" \
  --query 'LoadBalancers[0].DNSName' --output text)
echo "== 7) 테스트:"
echo "curl -I http://$ALB_DNS"

