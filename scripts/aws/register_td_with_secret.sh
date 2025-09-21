#!/usr/bin/env bash
set -euo pipefail

########################################
# === 기본 설정 (필요시만 바꿔도 됨) ===
########################################
REGION="ap-northeast-2"
CLUSTER="h2o2-cluster"
SERVICE="h2o2-service"
FAMILY="h2o2-task"
IMAGE_URI="037129617559.dkr.ecr.ap-northeast-2.amazonaws.com/h2o2/agent-app:prd"
CONTAINER_NAME="api"
APP_PORT=8501

# RDS/Secret 설정
DB_INSTANCE_ID="h2o2-db-prd"
DB_SECRET_NAME="h2o2-db-secrets"
DB_USER="ai"
DB_PASS="ai"
DB_NAME="ai"

export AWS_DEFAULT_REGION="$REGION"

echo "== Region: $REGION | Cluster: $CLUSTER | Service: $SERVICE"

########################################
# 0) RDS 엔드포인트 확인
########################################
RDS_ENDPOINT=$(aws rds describe-db-instances \
  --db-instance-identifier "$DB_INSTANCE_ID" \
  --query 'DBInstances[0].Endpoint.Address' --output text 2>/dev/null || true)

if [[ -z "${RDS_ENDPOINT:-}" || "$RDS_ENDPOINT" == "None" ]]; then
  echo "ERROR: RDS endpoint not found for DB instance '$DB_INSTANCE_ID'."
  echo " - 콘솔에서 RDS 인스턴스 상태 및 ID를 확인하세요."
  exit 1
fi
echo "RDS_ENDPOINT=$RDS_ENDPOINT"

DBURL="postgresql+psycopg://${DB_USER}:${DB_PASS}@${RDS_ENDPOINT}:5432/${DB_NAME}?sslmode=require"
echo "DATABASE_URL sample: ${DBURL%:*/*}/... (값은 Secrets로 주입됨)"

########################################
# 1) Secrets Manager: DATABASE_URL 생성/갱신
########################################
if aws secretsmanager describe-secret --secret-id "$DB_SECRET_NAME" >/dev/null 2>&1; then
  echo "Secret exists: $DB_SECRET_NAME → updating value"
  aws secretsmanager put-secret-value --secret-id "$DB_SECRET_NAME" \
    --secret-string "{\"DATABASE_URL\":\"$DBURL\"}" >/dev/null
else
  echo "Creating secret: $DB_SECRET_NAME"
  aws secretsmanager create-secret --name "$DB_SECRET_NAME" \
    --secret-string "{\"DATABASE_URL\":\"$DBURL\"}" \
    --description "H2O2 prod DB URL" >/dev/null
fi

DB_SECRET_ARN=$(aws secretsmanager describe-secret --secret-id "$DB_SECRET_NAME" --query 'ARN' --output text)
echo "DB_SECRET_ARN=$DB_SECRET_ARN"

########################################
# 2) 실행 역할(Execution Role) 보장
########################################
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ROLE_NAME="ecsTaskExecutionRole"

aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1 || \
aws iam create-role --role-name "$ROLE_NAME" \
  --assume-role-policy-document '{
    "Version":"2012-10-17",
    "Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]
  }' >/dev/null

aws iam attach-role-policy --role-name "$ROLE_NAME" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy >/dev/null 2>&1 || true

EXEC_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
echo "EXEC_ROLE_ARN=$EXEC_ROLE_ARN"

########################################
# 3) 로그 그룹 보장
########################################
aws logs create-log-group --log-group-name /ecs/h2o2 >/dev/null 2>&1 || true

########################################
# 4) 컨테이너 정의(JSON) — Secrets로 DATABASE_URL 주입
########################################
cat >/tmp/containers.json <<JSON
[{
  "name": "$CONTAINER_NAME",
  "image": "$IMAGE_URI",
  "command": ["/app/scripts/entrypoint.sh","serve"],
  "essential": true,
  "portMappings": [{ "containerPort": $APP_PORT, "protocol": "tcp" }],
  "logConfiguration": { "logDriver": "awslogs", "options": {
    "awslogs-group": "/ecs/h2o2",
    "awslogs-region": "$REGION",
    "awslogs-stream-prefix": "api"
  }},
  "environment": [
    { "name": "RUNTIME_ENV", "value": "prd" }
  ],
  "secrets": [
    { "name": "DATABASE_URL", "valueFrom": "$DB_SECRET_ARN:DATABASE_URL::" }
  ]
}]
JSON

########################################
# 5) Task Definition 등록
########################################
TD_NEW=$(aws ecs register-task-definition \
  --family "$FAMILY" \
  --requires-compatibilities FARGATE \
  --network-mode awsvpc \
  --cpu "1024" --memory "2048" \
  --execution-role-arn "$EXEC_ROLE_ARN" \
  --container-definitions file:///tmp/containers.json \
  --query 'taskDefinition.taskDefinitionArn' --output text)

if [[ -z "${TD_NEW:-}" || "$TD_NEW" == "None" ]]; then
  echo "ERROR: Failed to register task definition. Check previous output."
  exit 1
fi
echo "Registered TD: $TD_NEW"

########################################
# 6) 서비스 업데이트 (TG 매핑 유지)
########################################
SVC_STATUS=$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" \
  --query 'services[0].status' --output text 2>/dev/null || echo "NONE")
if [[ "$SVC_STATUS" != "ACTIVE" ]]; then
  echo "NOTE: Service '$SERVICE' is not ACTIVE (status=$SVC_STATUS)."
  echo " - ACTIVE일 때 아래 명령으로 적용하세요:"
  echo "aws ecs update-service --cluster $CLUSTER --service $SERVICE --task-definition $TD_NEW --force-new-deployment --health-check-grace-period-seconds 90"
  exit 0
fi

TG_ARN=$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" \
  --query 'services[0].loadBalancers[0].targetGroupArn' --output text 2>/dev/null || echo "")

if [[ -n "$TG_ARN" && "$TG_ARN" != "None" ]]; then
  aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" \
    --task-definition "$TD_NEW" \
    --load-balancers "targetGroupArn=$TG_ARN,containerName=$CONTAINER_NAME,containerPort=$APP_PORT" \
    --health-check-grace-period-seconds 90 \
    --force-new-deployment >/dev/null
else
  aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" \
    --task-definition "$TD_NEW" \
    --health-check-grace-period-seconds 90 \
    --force-new-deployment >/dev/null
fi

echo "Service updated. Verifying TD env/secrets..."
TD_NOW=$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --query 'services[0].taskDefinition' --output text)
aws ecs describe-task-definition --task-definition "$TD_NOW" \
  --query 'taskDefinition.containerDefinitions[0].{env:environment,secrets:secrets}' --output json
