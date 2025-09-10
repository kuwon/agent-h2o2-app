REGION="ap-northeast-2"
CLUSTER="h2o2-cluster"
SERVICE="h2o2-service"
FAMILY="h2o2-task"
IMAGE_URI="037129617559.dkr.ecr.ap-northeast-2.amazonaws.com/h2o2/agent-app:prd"
SECRET_NAME="h2o2-db-secrets"   # 콘솔에 보이는 시크릿 이름
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
EXEC_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/ecsTaskExecutionRole"
RDS_ENDPOINT=$(aws rds describe-db-instances --region $REGION \
  --db-instance-identifier h2o2-db-prd --query 'DBInstances[0].Endpoint.Address' --output text)

export AWS_DEFAULT_REGION="$REGION"

SECRET_ARN=$(aws secretsmanager describe-secret --secret-id "$SECRET_NAME" \
  --query 'ARN' --output text)

echo "SECRET_ARN=$SECRET_ARN"

# 순수 JSON만; 주석/트레일링 콤마 없음
cat > /tmp/containers.json <<'JSON'
[
  {
    "name": "api",
    "image": "037129617559.dkr.ecr.ap-northeast-2.amazonaws.com/h2o2/agent-app:prd",
    "command": ["/app/scripts/entrypoint.sh", "serve"],
    "workingDirectory": "/app",
    "essential": true,
    "portMappings": [
      { "containerPort": 8501, "protocol": "tcp" }
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/h2o2",
        "awslogs-region": "ap-northeast-2",
        "awslogs-stream-prefix": "api"
      }
    },
    "environment": [
      { "name": "RUNTIME_ENV", "value": "prd" },
      { "name": "PYTHONPATH", "value": "/app" }
    ],
    "secrets": [
      { "name": "DATABASE_URL", "valueFrom": "REPLACE_WITH_SECRET_ARN:DATABASE_URL::" },
      { "name": "DB_HOST", "valueFrom": "REPLACE_WITH_SECRET_ARN:DB_HOST::"  },
      { "name": "DB_PORT", "valueFrom": "REPLACE_WITH_SECRET_ARN:DB_PORT::" },
      { "name": "DB_USER", "valueFrom": "REPLACE_WITH_SECRET_ARN:DB_USER::" },
      { "name": "DB_PASS", "valueFrom": "REPLACE_WITH_SECRET_ARN:DB_PASS::" },
      { "name": "DB_NAME", "valueFrom": "REPLACE_WITH_SECRET_ARN:DB_NAME::" }
    ]
  }
]
JSON

# 시크릿 ARN 치환 (macOS)
sed -i '' "s#REPLACE_WITH_SECRET_ARN#$SECRET_ARN#g" /tmp/containers.json
# (Linux일 경우) sed -i "s#REPLACE_WITH_SECRET_ARN#$SECRET_ARN#g" /tmp/containers.json

# JSON 유효성 검사(권장)
jq . /tmp/containers.json >/dev/null

TD_NEW=$(aws ecs register-task-definition \
  --family "$FAMILY" \
  --requires-compatibilities FARGATE \
  --network-mode awsvpc \
  --cpu "1024" --memory "2048" \
  --execution-role-arn "$EXEC_ROLE_ARN" \
  --container-definitions file:///tmp/containers.json \
  --query 'taskDefinition.taskDefinitionArn' --output text)

echo "TD_NEW=$TD_NEW"

# 기존 TG 매핑 유지하면서 갱신
TG_ARN=$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" \
  --query 'services[0].loadBalancers[0].targetGroupArn' --output text 2>/dev/null || echo "")

if [ -n "$TG_ARN" ] && [ "$TG_ARN" != "None" ]; then
  aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" \
    --task-definition "$TD_NEW" \
    --load-balancers "targetGroupArn=$TG_ARN,containerName=api,containerPort=8501" \
    --health-check-grace-period-seconds 90 \
    --force-new-deployment
else
  aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" \
    --task-definition "$TD_NEW" \
    --health-check-grace-period-seconds 90 \
    --force-new-deployment
fi

TD_NOW=$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" \
  --query 'services[0].taskDefinition' --output text)

aws ecs describe-task-definition --task-definition "$TD_NOW" \
  --query 'taskDefinition.containerDefinitions[0].{wd:workingDirectory,env:environment,secrets:secrets}' --output json
# => secrets 배열에 DATABASE_URL, valueFrom에 시크릿 ARN+DATABASE_URL 키가 보여야 정상

