#!/usr/bin/env bash
set -euo pipefail

# ====== 환경 ======
REGION=${REGION:-ap-northeast-2}
CLUSTER=${CLUSTER:-h2o2-cluster}
SERVICE=${SERVICE:-h2o2-service}
LB_NAME=${LB_NAME:-h2o2-alb}
SVC_SG_NAME=${SVC_SG_NAME:-h2o2-svc-sg}
ALB_SG_NAME=${ALB_SG_NAME:-h2o2-alb-sg}

# 후보 포트(앱이 어디서 듣는지 모를 때 순차 프로빙)
CANDIDATE_PORTS=${CANDIDATE_PORTS:-"8501 8000 8080 80"}

echo "=== 0) 기본 정보 ==="
LB_ARN=$(aws elbv2 describe-load-balancers --region "$REGION" --names "$LB_NAME" \
  --query 'LoadBalancers[0].LoadBalancerArn' --output text)
ALB_DNS=$(aws elbv2 describe-load-balancers --region "$REGION" --load-balancer-arns "$LB_ARN" \
  --query 'LoadBalancers[0].DNSName' --output text)
VPC_ID=$(aws elbv2 describe-load-balancers --region "$REGION" --load-balancer-arns "$LB_ARN" \
  --query 'LoadBalancers[0].VpcId' --output text)
echo "ALB: $ALB_DNS"

SVC_SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
  --filters Name=group-name,Values="$SVC_SG_NAME" Name=vpc-id,Values="$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text)
ALB_SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
  --filters Name=group-name,Values="$ALB_SG_NAME" Name=vpc-id,Values="$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text)

LISTENER80=$(aws elbv2 describe-listeners --region "$REGION" --load-balancer-arn "$LB_ARN" \
  --query 'Listeners[?Port==`80`].ListenerArn' --output text)

# ====== 1) 서비스 상태/이벤트 확인 ======
echo "=== 1) ECS 서비스 상태 ==="
aws ecs describe-services --region "$REGION" --cluster "$CLUSTER" --services "$SERVICE" \
  --query 'services[0].{desired:desiredCount,running:runningCount,pending:pendingCount,events:events[:5].message}' --output json

# ====== 2) 실행 중 태스크 Public IP 획득 ======
echo "=== 2) 태스크 Public IP ==="
TASK=$(aws ecs list-tasks --region "$REGION" --cluster "$CLUSTER" --service-name "$SERVICE" \
  --query 'taskArns[0]' --output text)
if [[ -z "$TASK" || "$TASK" == "None" ]]; then
  echo "⚠️  실행 중 태스크가 없습니다. desiredCount를 1 이상으로 올리거나 배포를 다시 시도하세요."
  exit 1
fi
ENI=$(aws ecs describe-tasks --region "$REGION" --cluster "$CLUSTER" --tasks "$TASK" \
  --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text)
PUBIP=$(aws ec2 describe-network-interfaces --region "$REGION" --network-interface-ids "$ENI" \
  --query 'NetworkInterfaces[0].Association.PublicIp' --output text)
echo "TASK=$TASK | ENI=$ENI | PUBIP=$PUBIP"

# ====== 3) 내 IP를 서비스 SG에 임시 허용하고 포트 프로빙 ======
MYIP=$(curl -s https://checkip.amazonaws.com)/32
FOUND_PORT=""
OPENED_PORTS=()

for p in $CANDIDATE_PORTS; do
  # 임시 허용
  aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SVC_SG_ID" \
    --protocol tcp --port "$p" --cidr "$MYIP" >/dev/null 2>&1 || true
  OPENED_PORTS+=("$p")
  # 프로빙 (응답 코드 200-499면 통과로 간주)
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "http://$PUBIP:$p/")
  echo "Probe http://$PUBIP:$p/ -> $code"
  if [[ "$code" =~ ^[234][0-9][0-9]$ ]]; then
    FOUND_PORT="$p"
    break
  fi
done

if [[ -z "$FOUND_PORT" ]]; then
  echo "❌ 컨테이너가 후보 포트($CANDIDATE_PORTS)에서 응답하지 않습니다."
  echo " - 앱이 127.0.0.1로만 리슨하거나, 다른 포트일 가능성이 큽니다."
  echo " - 이미지의 실행 커맨드를 0.0.0.0:<port> 로 바꾸거나, CANDIDATE_PORTS를 확장해 재시도하세요."
  # 임시 규칙 정리
  for p in "${OPENED_PORTS[@]}"; do
    aws ec2 revoke-security-group-ingress --region "$REGION" --group-id "$SVC_SG_ID" \
      --protocol tcp --port "$p" --cidr "$MYIP" >/dev/null 2>&1 || true
  done
  exit 1
fi

echo "✅ 컨테이너 응답 포트 감지: $FOUND_PORT"

# 서비스 SG에 ALB SG -> FOUND_PORT 허용 (없으면 추가)
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SVC_SG_ID" \
  --ip-permissions "IpProtocol=tcp,FromPort=$FOUND_PORT,ToPort=$FOUND_PORT,UserIdGroupPairs=[{GroupId=$ALB_SG_ID}]" >/dev/null 2>&1 || true

# ====== 4) Target Group(FOUND_PORT)로 정렬 ======
# (있으면 재사용, 없으면 생성)
TG_NAME="h2o2-tg-$FOUND_PORT"
TG_ARN=$(aws elbv2 describe-target-groups --region "$REGION" --names "$TG_NAME" \
  --query 'TargetGroups[0].TargetGroupArn' --output text 2>/dev/null || echo "")
if [[ -z "$TG_ARN" || "$TG_ARN" == "None" ]]; then
  TG_ARN=$(aws elbv2 create-target-group --region "$REGION" \
    --name "$TG_NAME" --protocol HTTP --port "$FOUND_PORT" \
    --vpc-id "$VPC_ID" --target-type ip \
    --health-check-path "/" --health-check-port traffic-port \
    --matcher HttpCode=200-399 \
    --query 'TargetGroups[0].TargetGroupArn' --output text)
fi
# 리스너 80을 TG_ARN으로 전환
aws elbv2 modify-listener --region "$REGION" --listener-arn "$LISTENER80" \
  --default-actions "Type=forward,TargetGroupArn=$TG_ARN" >/dev/null

# ====== 5) ECS 서비스도 같은 TG/포트로 업데이트 + 재배포 ======
aws ecs update-service --region "$REGION" --cluster "$CLUSTER" --service "$SERVICE" \
  --load-balancers "targetGroupArn=$TG_ARN,containerName=api,containerPort=$FOUND_PORT" \
  --force-new-deployment >/dev/null

# ====== 6) 헬스 대기 ======
echo "헬스체크 대기중..."
for i in {1..24}; do
  states=$(aws elbv2 describe-target-health --region "$REGION" --target-group-arn "$TG_ARN" \
    --query 'TargetHealthDescriptions[].TargetHealth.State' --output text || true)
  echo "  [$i] $states"
  if echo "$states" | grep -q healthy; then
    echo "✅ Healthy!"
    break
  fi
  sleep 5
done

# ====== 7) 임시 규칙 정리 + 최종 테스트 ======
for p in "${OPENED_PORTS[@]}"; do
  aws ec2 revoke-security-group-ingress --region "$REGION" --group-id "$SVC_SG_ID" \
    --protocol tcp --port "$p" --cidr "$MYIP" >/dev/null 2>&1 || true
done

echo "ALB: http://$ALB_DNS"
curl -I "http://$ALB_DNS" || true

