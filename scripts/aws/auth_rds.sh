#!/usr/bin/env bash
set -euo pipefail

REGION="ap-northeast-2"
CLUSTER="h2o2-cluster"
SERVICE="h2o2-service"
RDS_ENDPOINT="h2o2-db-prd.c52yi26mgbl8.ap-northeast-2.rds.amazonaws.com"
OPEN_IF_MISSING=true   # 규칙 없으면 자동으로 열기

echo "[1] Get ECS task & ENI"
TASK_ARN=$(aws ecs list-tasks --region $REGION --cluster $CLUSTER --service-name $SERVICE --query 'taskArns[0]' --output text)
ENI_ID=$(aws ecs describe-tasks --region $REGION --cluster $CLUSTER --tasks "$TASK_ARN" \
--query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text)
ECS_SG=$(aws ec2 describe-network-interfaces --region $REGION --network-interface-ids "$ENI_ID" \
--query 'NetworkInterfaces[0].Groups[0].GroupId' --output text)
echo "TASK=$TASK_ARN ENI=$ENI_ID ECS_SG=$ECS_SG"

echo "[2] Find RDS SG by endpoint"
RDS_SG=$(aws rds describe-db-instances --region $REGION \
--query "DBInstances[?Endpoint.Address=='$RDS_ENDPOINT'].VpcSecurityGroups[0].VpcSecurityGroupId" --output text)
echo "RDS_SG=$RDS_SG"

echo "[3] Show current inbound on RDS SG (expect 5432 from ECS_SG)"
aws ec2 describe-security-groups --region $REGION --group-ids "$RDS_SG" \
--query 'SecurityGroups[0].IpPermissions'

NEED_OPEN=$(aws ec2 describe-security-groups --region $REGION --group-ids "$RDS_SG" \
--query "SecurityGroups[0].IpPermissions[?FromPort==\`5432\` && ToPort==\`5432\` && IpProtocol==\`tcp\` && contains(UserIdGroupPairs[].GroupId, \`$ECS_SG\`)] | length(@)" --output text)

if [[ "$NEED_OPEN" == "0" && "$OPEN_IF_MISSING" == "true" ]]; then
  echo "[4] Opening inbound 5432 from ECS_SG to RDS_SG..."
  aws ec2 authorize-security-group-ingress --region $REGION --group-id "$RDS_SG" \
    --protocol tcp --port 5432 --source-group "$ECS_SG" || true
else
  echo "[4] Inbound already present or auto-open disabled."
fi

echo "[5] Quick TCP check from local (optional; replace with ECS container check if needed)"
# Requires your local to reach RDS; for ECS container check, run /dev/tcp test inside container.