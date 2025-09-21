#!/usr/bin/env bash
set -euo pipefail

# ==== 고정값 ====
REGION=ap-northeast-2
CLUSTER=h2o2-cluster
SERVICE=h2o2-service
FAMILY=h2o2-task
APP_PORT=8501

# 네 퍼블릭 서브넷 2개 (ALB/태스크용)
SUBNETS_PUBLIC="subnet-095f958fbf81ed9d4 subnet-08e75869bc55b5df1"

export AWS_DEFAULT_REGION="$REGION"

# 0) 서비스 링크드 롤 & 클러스터 보장
aws iam create-service-linked-role --aws-service-name ecs.amazonaws.com >/dev/null 2>&1 || true
STATUS=$(aws ecs describe-clusters --clusters "$CLUSTER" --query 'clusters[0].status' --output text 2>/dev/null || echo "NONE")
if [[ "$STATUS" != "ACTIVE" ]]; then
  aws ecs create-cluster --cluster-name "$CLUSTER" >/dev/null
fi

# 1) 최신/유효 TD 추출 (너가 확인했던 h2o2-task:5가 있으면 그걸, 없으면 최신)
TD_OK=$(aws ecs list-task-definitions --family-prefix "$FAMILY" --status ACTIVE --sort DESC \
  --max-items 1 --query 'taskDefinitionArns[0]' --output text)
aws ecs describe-task-definition --task-definition "$TD_OK" \
  --query 'taskDefinition.status' --output text | grep -q ACTIVE
CN=$(aws ecs describe-task-definition --task-definition "$TD_OK" \
  --query 'taskDefinition.containerDefinitions[0].name' --output text)

# 2) SG/VPC/TG 확보
FIRST_SUBNET=$(echo $SUBNETS_PUBLIC | awk '{print $1}')
VPC_ID=$(aws ec2 describe-subnets --subnet-ids "$FIRST_SUBNET" --query 'Subnets[0].VpcId' --output text)

# Service SG (h2o2-svc-sg) 조회
SVC_SG_ID=$(aws ec2 describe-security-groups \
  --filters Name=group-name,Values=h2o2-svc-sg Name=vpc-id,Values="$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text)

# ALB/TG 확보(기존 TG 재사용 또는 생성)
LB_ARN=$(aws elbv2 describe-load-balancers --names h2o2-alb \
  --query 'LoadBalancers[0].LoadBalancerArn' --output text 2>/dev/null || echo "")
if [[ -z "$LB_ARN" || "$LB_ARN" == "None" ]]; then
  echo "ERROR: ALB 'h2o2-alb' not found. 먼저 ALB를 만들어 주세요."; exit 1
fi

TG_ARN=$(aws elbv2 describe-target-groups --names h2o2-tg-8501 \
  --query 'TargetGroups[0].TargetGroupArn' --output text 2>/dev/null || echo "")
if [[ -z "$TG_ARN" || "$TG_ARN" == "None" ]]; then
  TG_ARN=$(aws elbv2 create-target-group \
    --name h2o2-tg-8501 --protocol HTTP --port $APP_PORT \
    --vpc-id "$VPC_ID" --target-type ip \
    --health-check-path "/" --matcher HttpCode=200-399 \
    --query 'TargetGroups[0].TargetGroupArn' --output text)
fi

# 리스너 80이 위 TG로 포워딩하도록 보정
LISTENER80=$(aws elbv2 describe-listeners --load-balancer-arn "$LB_ARN" \
  --query 'Listeners[?Port==`80`].ListenerArn' --output text)
aws elbv2 modify-listener --listener-arn "$LISTENER80" \
  --default-actions Type=forward,TargetGroupArn="$TG_ARN" >/dev/null

# 3) 서비스 상태 점검
SVC_STATUS=$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" \
  --query 'services[0].status' --output text 2>/dev/null || echo "NONE")

SUBNETS_CSV=$(echo "$SUBNETS_PUBLIC" | sed 's/ /,/g')

if [[ "$SVC_STATUS" == "ACTIVE" ]]; then
  # 업데이트 경로
  aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" \
    --task-definition "$TD_OK" \
    --load-balancers targetGroupArn="$TG_ARN",containerName="$CN",containerPort=$APP_PORT \
    --health-check-grace-period-seconds 90 \
    --force-new-deployment
else
  # 생성 경로 (INACTIVE/DRAINING/NONE 모두 새로 생성)
  aws ecs create-service \
    --cluster "$CLUSTER" --service-name "$SERVICE" \
    --task-definition "$TD_OK" \
    --desired-count 1 --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS_CSV],securityGroups=[$SVC_SG_ID],assignPublicIp=ENABLED}" \
    --load-balancers "targetGroupArn=$TG_ARN,containerName=$CN,containerPort=$APP_PORT" \
    --health-check-grace-period-seconds 90
fi

# 4) 상태 출력
aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" \
  --query 'services[0].{status:status,desired:desiredCount,running:runningCount,pending:pendingCount,taskDefinition:taskDefinition}' --output json

# TG 헬스 확인
aws elbv2 describe-target-health --target-group-arn "$TG_ARN" \
  --query 'TargetHealthDescriptions[].{IP:Target.Id,State:TargetHealth.State,Reason:TargetHealth.Reason}' --output table
