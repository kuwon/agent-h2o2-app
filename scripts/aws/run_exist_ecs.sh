#!/usr/bin/env bash
set -euo pipefail

REGION=ap-northeast-2
CLUSTER=h2o2-cluster
SERVICE=h2o2-service
IMAGE_URI="037129617559.dkr.ecr.ap-northeast-2.amazonaws.com/h2o2/agent-app:prd"
APP_PORT=8501

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
EXEC_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/ecsTaskExecutionRole"

aws logs create-log-group --region "$REGION" --log-group-name /ecs/h2o2 >/dev/null 2>&1 || true

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
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/h2o2",
        "awslogs-region": "$REGION",
        "awslogs-stream-prefix": "api"
      }
    },
    "environment": [
      { "name": "DATABASE_URL", "value": "postgresql+psycopg://<user>:<pass>@<rds-endpoint>:5432/<db>" }
    ]
  }]
}
JSON

TD_ARN=$(aws ecs register-task-definition --region "$REGION" \
  --cli-input-json file:///tmp/td.json \
  --query 'taskDefinition.taskDefinitionArn' --output text)

aws ecs update-service --region "$REGION" \
  --cluster "$CLUSTER" --service "$SERVICE" \
  --task-definition "$TD_ARN" --force-new-deployment
