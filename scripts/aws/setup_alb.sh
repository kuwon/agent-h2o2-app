REGION="ap-northeast-2"
CLUSTER="h2o2-cluster"
SERVICE="h2o2-service"

TG_8501_ARN="arn:aws:elasticloadbalancing:ap-northeast-2:037129617559:targetgroup/h2o2-tg-8501/880182c2be427516"
TG_8000_ARN="arn:aws:elasticloadbalancing:ap-northeast-2:037129617559:targetgroup/h2o2-tg-8000/61d77685bfc3fab3"

aws ecs update-service \
  --region "$REGION" \
  --cluster "$CLUSTER" \
  --service "$SERVICE" \
  --load-balancers \
    targetGroupArn="$TG_8501_ARN",containerName="api",containerPort=8501 \
    targetGroupArn="$TG_8000_ARN",containerName="api",containerPort=8000 \
  --force-new-deployment


aws elbv2 modify-target-group \
  --region "$REGION" \
  --target-group-arn "$TG_8501_ARN" \
  --health-check-path "/_stcore/health" \
  --matcher HttpCode=200-399

aws elbv2 modify-target-group \
  --region "$REGION" \
  --target-group-arn "$TG_8000_ARN" \
  --health-check-path "/v1/health" \
  --matcher HttpCode=200-399
