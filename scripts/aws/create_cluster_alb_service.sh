#!/usr/bin/env bash
set -euo pipefail

# ==== 환경 ====
REGION=ap-northeast-2
CLUSTER=h2o2-cluster
SERVICE=h2o2-service
LB_NAME=h2o2-alb
APP_PORT=8501
IMAGE_URI="037129617559.dkr.ecr.ap-northeast-2.amazonaws.com/h2o2/agent-app:prd"

# 너의 퍼블릭 서브넷 2개 (ALB 용) — 너가 준 값 그대로
SUBNETS_PUBLIC="subnet-095f958fbf81ed9d4 subnet-08e75869bc55b5df1"
# 태스크도 일단 퍼블릭에서 테스트(운영 전환 시 프라이빗 권장)
SUBNETS_TASK="$SUBNETS_PUBLIC"

# ==== 0) 리전 고정 (실수 방지) ====
export AWS_DEFAULT_REGION="$REGION"

# ==== 1) ECS 서비스 링크드 롤 ====
aws iam create-service-linked-role --aws-service-name ecs.amazonaws.com >/dev/null 2>&1 || true

# ==== 2) 클러스터 보장(ACTIVE) ====
STATUS=$(aws ecs describe-clusters --clusters "$CLUSTER" --query 'clusters[0].status' --output text 2>/dev/null || echo "NONE")
if [[ "$STATUS" != "ACTIVE" ]]; then
  echo "Cluster '$CLUSTER' status=$STATUS -> creating new..."
  aws ecs create-cluster --cluster-name "$CLUSTER" >/dev/null
fi
# 대기 (ACTIVE 될 때까지)
for i in {1..20}; do
  STATUS=$(aws ecs describe-clusters --clusters "$CLUSTER" --query 'clusters[0].status' --output text)
  echo "Cluster status: $STATUS"
  [[ "$STATUS" == "ACTIVE" ]] && break
  sleep 2
done
[[ "$STATUS" == "ACTIVE" ]] || { echo "Cluster not ACTIVE"; exit 1; }

# ==== 3) VPC/SG ====
FIRST_SUBNET=$(echo $SUBNETS_PUBLIC | awk '{print $1}')
VPC_ID=$(aws ec2 describe-subnets --subnet-ids "$FIRST_SUBNET" --query 'Subnets[0].VpcId' --output text)

# ALB SG
ALB_SG_ID=$(aws ec2 describe-security-groups \
  --filters Name=group-name,Values=h2o2-alb-sg Name=vpc-id,Values="$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text)
if [[ "$ALB_SG_ID" == "None" || -z "$ALB_SG_ID" ]]; then
  ALB_SG_ID=$(aws ec2 create-security-group --group-name h2o2-alb-sg --description "ALB SG" --vpc-id "$VPC_ID" --query GroupId --output text)
  aws ec2 authorize-security-group-ingress --group-id "$ALB_SG_ID" \
    --ip-permissions IpProtocol=tcp,FromPort=80,ToPort=80,IpRanges='[{CidrIp=0.0.0.0/0}]' >/dev/null
fi

# Service SG
SVC_SG_ID=$(aws ec2 describe-security-groups \
  --filters Name=group-name,Values=h2o2-svc-sg Name=vpc-id,Values="$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text)
if [[ "$SVC_SG_ID" == "None" || -z "$SVC_SG_ID" ]]; then
  SVC_SG_ID=$(aws ec2 create-security-group --group-name h2o2-svc-sg --description "Service SG" --vpc-id "$VPC_ID" --query GroupId --output text)
  aws ec2 authorize-security-group-ingress --group-id "$SVC_SG_ID" \
    --ip-permissions IpProtocol=tcp,FromPort="$APP_PORT",ToPort="$APP_PORT",UserIdGroupPairs="[{GroupId=$ALB_SG_ID}]" >/dev/null
fi

# ==== 4) ALB / TG / Listener ====
LB_ARN=$(aws elbv2 describe-load-balancers --names "$LB_NAME" --query 'LoadBalancers[0].LoadBalancerArn' --output text 2>/dev/null || echo "")
if [[ -z "$LB_ARN" || "$LB_ARN" == "None" ]]; then
  LB_ARN=$(aws elbv2 create-load-balancer \
    --name "$LB_NAME" --type application --scheme internet-facing \
    --subnets $SUBNETS_PUBLIC --security-groups "$ALB_SG_ID" \
    --query 'LoadBalancers[0].LoadBalancerArn' --output text)
fi
aws elbv2 wait load-balancer-available --load-balancer-arns "$LB_ARN"

TG_NAME="h2o2-tg-$APP_PORT"
TG_ARN=$(aws elbv2 describe-target-groups --names "$TG_NAME" --query 'TargetGroups[0].TargetGroupArn' --output text 2>/dev/null || echo "")
if [[ -z "$TG_ARN" || "$TG_ARN" == "None" ]]; then
  TG_ARN=$(aws elbv2 create-target-group \
    --name "$TG_NAME" --protocol HTTP --port "$APP_PORT" \
    --vpc-id "$VPC_ID" --target-type ip \
    --health-check-protocol HTTP --health-check-port traffic-port --health-check-path / \
    --matcher HttpCode=200-399 \
    --query 'TargetGroups[0].TargetGroupArn' --output text)
fi

LISTENER80=$(aws elbv2 describe-listeners --load-balancer-arn "$LB_ARN" \
  --query 'Listeners[?Port==`80`].ListenerArn' --output text)
if [[ -z "$LISTENER80" || "$LISTENER80" == "None" ]]; then
  LISTENER80=$(aws elbv2 create-listener --load-balancer-arn "$LB_ARN" \
    --protocol HTTP --port 80 \
    --default-actions Type=forward,TargetGroupArn="$TG_ARN" \
    --query 'Listeners[0].ListenerArn' --output text)
else
  aws elbv2 modify-listener --listener-arn "$LISTENER80" \
    --default-actions Type=forward,TargetGroupArn="$TG_ARN" >/dev/null
fi

# ==== 5) Task Definition ====
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
EXEC_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/ecsTaskExecutionRole"
aws logs create-log-group --log-group-name /ecs/h2o2 >/dev/null 2>&1 || true

cat >/tmp/td.json <<JSON
{
  "family": "h2o2-task",
  "requiresCompatibilities": ["FARGATE"],
  "networkMode": "awsvpc",
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "$EXEC_ROLE_ARN",
  "containerDefinitions": [{
    "name": "api",
    "image": "$IMAGE_URI",
    "portMappings": [{ "containerPort": $APP_PORT, "protocol": "tcp" }],
    "essential": true,
    "logConfiguration": { "logDriver": "awslogs", "options": {
      "awslogs-group": "/ecs/h2o2", "awslogs-region": "$REGION", "awslogs-stream-prefix": "api"
    }}
  }]
}
JSON

TD_ARN=$(aws ecs register-task-definition --cli-input-json file:///tmp/td.json \
  --query 'taskDefinition.taskDefinitionArn' --output text)

# ==== 6) 서비스 생성/업데이트 ====
SUBNETS_TASK_CSV=$(echo "$SUBNETS_TASK" | sed 's/ /,/g')
SVC_STATUS=$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --query 'services[0].status' --output text 2>/dev/null || echo "")
if [[ "$SVC_STATUS" == "ACTIVE" ]]; then
  aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" \
    --task-definition "$TD_ARN" \
    --load-balancers "targetGroupArn=$TG_ARN,containerName=api,containerPort=$APP_PORT" \
    --force-new-deployment >/dev/null
else
  aws ecs create-service \
    --cluster "$CLUSTER" --service-name "$SERVICE" \
    --task-definition "$TD_ARN" --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS_TASK_CSV],securityGroups=[$SVC_SG_ID],assignPublicIp=ENABLED}" \
    --load-balancers "targetGroupArn=$TG_ARN,containerName=api,containerPort=$APP_PORT" >/dev/null
fi

ALB_DNS=$(aws elbv2 describe-load-balancers --load-balancer-arns "$LB_ARN" --query 'LoadBalancers[0].DNSName' --output text)
echo "Open: http://$ALB_DNS"
