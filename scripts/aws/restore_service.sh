REGION=ap-northeast-2
CLUSTER=h2o2-cluster
SERVICE=h2o2-service
FAMILY=h2o2-task
APP_PORT=8501
SUBNETS_PUBLIC="subnet-095f958fbf81ed9d4 subnet-08e75869bc55b5df1"

export AWS_DEFAULT_REGION=$REGION

# 최신 ACTIVE TD 하나 집기
TD_OK=$(aws ecs list-task-definitions --family-prefix "$FAMILY" --status ACTIVE --sort DESC \
  --max-items 1 --query 'taskDefinitionArns[0]' --output text)
CN=$(aws ecs describe-task-definition --task-definition "$TD_OK" \
  --query 'taskDefinition.containerDefinitions[0].name' --output text)

# VPC/SG/TG
FIRST_SUBNET=$(echo $SUBNETS_PUBLIC | awk '{print $1}')
VPC_ID=$(aws ec2 describe-subnets --region $REGION --subnet-ids "$FIRST_SUBNET" --query 'Subnets[0].VpcId' --output text)
SVC_SG_ID=$(aws ec2 describe-security-groups --region $REGION \
  --filters Name=group-name,Values=h2o2-svc-sg Name=vpc-id,Values="$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text)

LB_ARN=$(aws elbv2 describe-load-balancers --region $REGION --names h2o2-alb \
  --query 'LoadBalancers[0].LoadBalancerArn' --output text)
LISTENER80=$(aws elbv2 describe-listeners --region $REGION --load-balancer-arn "$LB_ARN" \
  --query 'Listeners[?Port==`80`].ListenerArn' --output text)

TG_ARN=$(aws elbv2 describe-target-groups --region $REGION --names h2o2-tg-8501 \
  --query 'TargetGroups[0].TargetGroupArn' --output text 2>/dev/null || echo "")
if [[ -z "$TG_ARN" || "$TG_ARN" == "None" ]]; then
  TG_ARN=$(aws elbv2 create-target-group --region $REGION \
    --name h2o2-tg-8501 --protocol HTTP --port $APP_PORT --vpc-id "$VPC_ID" --target-type ip \
    --health-check-path "/" --matcher HttpCode=200-399 \
    --query 'TargetGroups[0].TargetGroupArn' --output text)
fi
aws elbv2 modify-listener --region $REGION --listener-arn "$LISTENER80" \
  --default-actions Type=forward,TargetGroupArn="$TG_ARN" >/dev/null

# 서비스 상태에 따라 업데이트/생성
SVC_STATUS=$(aws ecs describe-services --region $REGION --cluster "$CLUSTER" --services "$SERVICE" \
  --query 'services[0].status' --output text 2>/dev/null || echo "NONE")
SUBNETS_CSV=$(echo "$SUBNETS_PUBLIC" | sed 's/ /,/g')

if [[ "$SVC_STATUS" == "ACTIVE" ]]; then
  aws ecs update-service --region $REGION --cluster "$CLUSTER" --service "$SERVICE" \
    --task-definition "$TD_OK" \
    --load-balancers targetGroupArn="$TG_ARN",containerName="$CN",containerPort=$APP_PORT \
    --desired-count 1 \
    --health-check-grace-period-seconds 90 \
    --force-new-deployment
else
  aws ecs create-service --region $REGION \
    --cluster "$CLUSTER" --service-name "$SERVICE" \
    --task-definition "$TD_OK" \
    --desired-count 1 --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS_CSV],securityGroups=[$SVC_SG_ID],assignPublicIp=ENABLED}" \
    --load-balancers "targetGroupArn=$TG_ARN,containerName=$CN,containerPort=$APP_PORT" \
    --health-check-grace-period-seconds 90
fi

# 상태/헬스 확인
aws ecs describe-services --region $REGION --cluster "$CLUSTER" --services "$SERVICE" \
  --query 'services[0].{status:status,desired:desiredCount,running:runningCount,pending:pendingCount,td:taskDefinition}' --output json
aws elbv2 describe-target-health --region $REGION --target-group-arn "$TG_ARN" \
  --query 'TargetHealthDescriptions[].{IP:Target.Id,State:TargetHealth.State,Desc:TargetHealth.Description}' --output table
