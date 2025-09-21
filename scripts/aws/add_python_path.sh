#!/usr/bin/env bash
set -euo pipefail

REGION="ap-northeast-2"
CLUSTER="h2o2-cluster"
SERVICE="h2o2-service"
FAMILY="h2o2-task"
IMAGE_URI="037129617559.dkr.ecr.ap-northeast-2.amazonaws.com/h2o2/agent-app:prd"
CONTAINER_NAME="api"
APP_PORT=8501
DB_SECRET_NAME="h2o2-db-secrets"

export AWS_DEFAULT_REGION="$REGION"

DB_SECRET_ARN=$(aws secretsmanager describe-secret --secret-id "$DB_SECRET_NAME" --query 'ARN' --output text)
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
EXEC_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/ecsTaskExecutionRole"

aws logs create-log-group --log-group-name /ecs/h2o2 >/dev/null 2>&1 || true

# 컨테이너 정의 - 핵심: workingDirectory & PYTHONPATH
cat >/tmp/containers.json <<JSON
[{
  "name": "$CONTAINER_NAME",
  "image": "$IMAGE_URI",
  "command": ["/app/scripts/entrypoint.sh","serve"],
  "workingDirectory": "/app",
  "essential": true,
  "portMappings": [{ "containerPort": $APP_PORT, "protocol": "tcp" }],
  "logConfiguration": { "logDriver": "awslogs", "options": {
    "awslogs-group": "/ecs/h2o2",
    "awslogs-region": "$REGION",
    "awslogs-stream-prefix": "api"
  }},
  "environment": [
    { "name": "RUNTIME_ENV", "value": "prd" },
    { "name": "PYTHONPATH", "value": "/app" }
  ],
  "secrets": [
    { "name": "DATABASE_URL", "valueFrom": "$DB_SECRET_ARN:DATABASE_URL::" }
  ]
}]
JSON

TD_NEW=$(aws ecs register-task-definition \
  --family "$FAMILY" \
  --requires-compatibilities FARGATE \
  --network-mode awsvpc \
  --cpu "1024" --memory "2048" \
  --execution-role-arn "$EXEC_ROLE_ARN" \
  --container-definitions file:///tmp/containers.json \
  --query 'taskDefinition.taskDefinitionArn' --output text)
echo "Registered TD: $TD_NEW"

# 서비스에 새 TD 적용(강제 롤링)
TG_ARN=$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" \
  --query 'services[0].loadBalancers[0].targetGroupArn' --output text 2>/dev/null || echo "")
if [ -n "$TG_ARN" ] && [ "$TG_ARN" != "None" ]; then
  aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" \
    --task-definition "$TD_NEW" \
    --load-balancers "targetGroupArn=$TG_ARN,containerName=$CONTAINER_NAME,containerPort=$APP_PORT" \
    --health-check-grace-period-seconds 90 \
    --force-new-deployment
else
  aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" \
    --task-definition "$TD_NEW" \
    --health-check-grace-period-seconds 90 \
    --force-new-deployment
fi
