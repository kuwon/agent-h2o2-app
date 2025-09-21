REGION=ap-northeast-2
CLUSTER=h2o2-cluster
SERVICE=h2o2-service

# 1) 서비스 이벤트(최근 실패 이유가 바로 찍힘)
aws ecs describe-services --region $REGION --cluster $CLUSTER --services $SERVICE \
  --query 'services[0].events[0:10].[createdAt,message]' --output table

# 2) 최신 태스크 상태/종료 사유
TASK_ARN=$(aws ecs list-tasks --region $REGION --cluster $CLUSTER --service-name $SERVICE \
  --desired-status RUNNING --query 'taskArns[0]' --output text)
if [ "$TASK_ARN" = "None" ] || [ -z "$TASK_ARN" ]; then
  TASK_ARN=$(aws ecs list-tasks --region $REGION --cluster $CLUSTER --service-name $SERVICE \
    --desired-status STOPPED --query 'taskArns[0]' --output text)
fi

aws ecs describe-tasks --region $REGION --cluster $CLUSTER --tasks "$TASK_ARN" \
  --query 'tasks[0].{lastStatus:lastStatus,stopCode:stopCode,stoppedReason:stoppedReason,containers:containers[].{name:name,lastStatus:lastStatus,reason:reason,exitCode:exitCode}}' \
  --output json

# 3) 컨테이너 이미지/포트 확인(배포된 값과 앱의 실제 리슨 포트 비교)
aws ecs describe-tasks --region $REGION --cluster $CLUSTER --tasks "$TASK_ARN" \
  --query 'tasks[0].containers[].{name:name,image:image,portMappings:networkBindings}'

# 4) 로드밸런서/TG 상태와 헬스 이유
TG_ARN=$(aws ecs describe-services --region $REGION --cluster $CLUSTER --services $SERVICE \
  --query 'services[0].loadBalancers[0].targetGroupArn' --output text)
aws elbv2 describe-target-health --target-group-arn "$TG_ARN" \
  --query 'TargetHealthDescriptions[].{Id:Target.Id,Port:Target.Port,State:TargetHealth.State,Reason:TargetHealth.Reason,Desc:TargetHealth.Description}' \
  --output table

# 5) TG 설정(헬스체크 경로/포트/성공코드)
aws elbv2 describe-target-groups --target-group-arns "$TG_ARN" \
  --query 'TargetGroups[0].{Proto:Protocol,Port:Port,TargetType:TargetType,Matcher:Matcher.HttpCode,HealthCheckPath:HealthCheckPath,HealthPort:HealthCheckPort}' \
  --output table

# 6) 로그 그룹 이름(CloudWatch Logs) 자동 추출 후 최근 로그 tail (awslogs 사용 시)
TD_ARN=$(aws ecs describe-services --region $REGION --cluster $CLUSTER --services $SERVICE \
  --query 'services[0].taskDefinition' --output text)
aws ecs describe-task-definition --region $REGION --task-definition "$TD_ARN" --query \
  'taskDefinition.containerDefinitions[].logConfiguration.options."awslogs-group"' --output text

# 위에서 나온 로그 그룹으로 확인(예)
# aws logs tail /ecs/h2o2-task/app --since 15m --follow
