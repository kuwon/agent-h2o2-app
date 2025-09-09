#!/usr/bin/env bash
set -euo pipefail

REGION=ap-northeast-2
CLUSTER=h2o2-cluster
SERVICE=h2o2-service
APP_PORT=8501   # ← 네 앱 포트로 맞춰줘
IMAGE_URI="037129617559.dkr.ecr.ap-northeast-2.amazonaws.com/h2o2/agent-app:prd"

# 네가 준 퍼블릭 서브넷 2개
SUBNETS_PUBLIC="subnet-095f958fbf81ed9d4 subnet-08e75869bc55b5df1"
# 태스크도 임시로 퍼블릭 서브넷 사용
SUBNETS_TASK="$SUBNETS_PUBLIC"

# VPC 추출(첫 퍼블릭 서브넷 기준)
FIRST_SUBNET=$(echo $SUBNETS_PUBLIC | awk '{print $1}')
VPC_ID=$(aws ec2 describe-subnets --region "$REGION" --subnet-ids "$FIRST_SUBNET" \
  --query 'Subnets[0].VpcId' --output text)

# 1) ECS SLR 문제 예방(있으면 패스)
aws iam create-service-linked-role --aws-service-name ecs.amazonaws.com >/dev/null 2>&1 || true

# 2) 클러스터(있으면 패스)
aws ecs describe-clusters --region "$REGION" --clusters "$CLUSTER" \
  --query 'clusters[0].status' --output text | grep -q ACTIVE \
  || aws ecs create-cluster --region "$REGION" --cluster-name "$CLUSTER" >/dev/null

# 3) 실행 역할(있으면 패스)
aws iam get-role --role-name ecsTaskExecutionRole >/dev/null 2>&1 || {
  aws iam create-role --role-name ecsTaskExecutionRole \
    --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]
    }' >/dev/null
  aws iam attach-role-policy --role-name ecsTaskExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy >/dev/null
}

# 4) 보안그룹: ALB(80 오픈), 서비스(앱포트는 ALB SG만 허용)
ALB_SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
  --filters Name=group-name,Values=h2o2-alb-sg Name=vpc-id,Values="$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text)
if [[ "$ALB_SG_ID" == "None" || -z "$ALB_SG_ID" ]]; then
  ALB_SG_ID=$(aws ec2 create-security-group --region "$REGION" \
    --group-name h2o2-alb-sg --description "ALB SG" --vpc-id "$VPC_ID" \
    --query GroupId --output text)
  aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$ALB_SG_ID" \
    --ip-permissions IpProtocol=tcp,FromPort=80,ToPort=80,IpRanges='[{CidrIp=0.0.0.0/0}]' >/dev/null
fi

SVC_SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
  --filters Name=group-name,Values=h2o2-svc-sg Name=vpc-id,Values="$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text)
if [[ "$SVC_SG_ID" == "None" || -z "$SVC_SG_ID" ]]; then
  SVC_SG_ID=$(aws ec2 create-security-group --region "$REGION" \
    --group-name h2o2-svc-sg --description "Service SG" --vpc-id "$VPC_ID" \
    --query GroupId --output text)
  aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SVC_SG_ID" \
    --ip-permissions IpProtocol=tcp,FromPort="$APP_PORT",ToPort="$APP_PORT",UserIdGroupPairs="[{GroupId=$ALB_SG_ID}]" >/dev/null
fi

# 5) ALB / TG / 리스너
SUBNETS_PUBLIC_CSV=$(echo "$SUBNETS_PUBLIC" | sed 's/ /,/g')
LB_ARN=$(aws elbv2 create-load-balancer --region "$REGION" \
  --name h2o2-alb --type application --scheme internet-facing \
  --subnets $SUBNETS_PUBLIC \
  --security-groups "$ALB_SG_ID" \
  --query 'LoadBalancers[0].LoadBalancerArn' --output text 2>/dev/null \
  || aws elbv2 describe-load-balancers --region "$REGION" \
       --names h2o2-alb --query 'LoadBalancers[0].LoadBalancerArn' --output text)

TG_ARN=$(aws elbv2 create-target-group --region "$REGION" \
  --name h2o2-tg --protocol HTTP --port "$APP_PORT" \
  --vpc-id "$VPC_ID" --target-type ip \
  --health-check-protocol HTTP --health-check-port traffic-port --health-check-path / \
  --query 'TargetGroups[0].TargetGroupArn' --output text 2>/dev/null \
  || aws elbv2 describe-target-groups --region "$REGION" \
       --names h2o2-tg --query 'TargetGroups[0].TargetGroupArn' --output text)

# HTTP 80 리스너(있으면 패스)
aws elbv2 describe-listeners --region "$REGION" --load-balancer-arn "$LB_ARN" \
  --query 'Listeners[?Port==`80`].ListenerArn' --output text | grep -q arn: \
  || aws elbv2 create-listener --region "$REGION" \
       --load-balancer-arn "$LB_ARN" --protocol HTTP --port 80 \
       --default-actions Type=forward,TargetGroupArn="$TG_ARN" >/dev/null

# 6) 태스크 정의
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
EXEC_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/ecsTaskExecutionRole"
aws logs create-log-group --region "$REGION" --log-group-name /ecs/h2o2 >/dev/null 2>&1 || true

cat >/tmp/h2o2-td.json <<JSON
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
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/h2o2",
        "awslogs-region": "$REGION",
        "awslogs-stream-prefix": "api"
      }
    }
  }]
}
JSON

TD_ARN=$(aws ecs register-task-definition --region "$REGION" \
  --cli-input-json file:///tmp/h2o2-td.json \
  --query 'taskDefinition.taskDefinitionArn' --output text)

# 7) 서비스 생성(태스크도 퍼블릭 서브넷 + 퍼블릭 IP 할당)
SUBNETS_TASK_CSV=$(echo "$SUBNETS_TASK" | sed 's/ /,/g')
aws ecs describe-services --region "$REGION" --cluster "$CLUSTER" --services "$SERVICE" \
  --query 'services[0].status' --output text | grep -q ACTIVE || \
aws ecs create-service --region "$REGION" \
  --cluster "$CLUSTER" --service-name "$SERVICE" \
  --task-definition "$TD_ARN" --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS_TASK_CSV],securityGroups=[$SVC_SG_ID],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=$TG_ARN,containerName=api,containerPort=$APP_PORT" >/dev/null

# 8) 접속 주소 출력
ALB_DNS=$(aws elbv2 describe-load-balancers --region "$REGION" \
  --load-balancer-arns "$LB_ARN" --query 'LoadBalancers[0].DNSName' --output text)
echo "Open: http://$ALB_DNS"

