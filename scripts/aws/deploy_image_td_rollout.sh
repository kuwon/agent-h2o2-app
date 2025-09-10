#!/usr/bin/env bash
set -euo pipefail

### =========================
### 기본 설정(환경에 맞게 수정)
### =========================
REGION="ap-northeast-2"
CLUSTER="h2o2-cluster"
SERVICE="h2o2-service"
FAMILY="h2o2-task"

# ECR
IMAGE_BASE_URI="037129617559.dkr.ecr.ap-northeast-2.amazonaws.com/h2o2/agent-app"
ECR_REPO_NAME="h2o2/agent-app"

# ALB / TG
ALB_LISTENER_ARN=""   # 예: arn:aws:elasticloadbalancing:...:listener/app/xxx/yyy/zzz (없으면 기본 규칙 갱신 생략)
RULE_ARN=""           # 특정 경로/호스트 규칙을 갱신하려면 지정. 지정 시 LISTENER_ARN 대신 이 값을 사용
VPC_ID=""             # 새 TG를 생성해야 할 때 필요 (기존 서비스에 TG가 이미 있으면 생략 가능)

# 헬스체크/포트/TG 설정(필요 시 수정)
TG_NAME=""            # 새 TG를 생성할 때 사용할 이름(미지정 시 SERVICE 기반 자동명)
TG_TARGET_TYPE="ip"   # Fargate는 ip 권장 (ec2의 인스턴스 타겟이면 instance)
TG_PROTOCOL="HTTP"
TG_HEALTH_PROTOCOL="HTTP"
TG_HEALTH_PATH="/"    # FastAPI면 /health 등
TG_HEALTH_PORT="traffic-port"
TG_HEALTHY_THRESHOLD=2
TG_UNHEALTHY_THRESHOLD=2
TG_HEALTH_INTERVAL=15
TG_HEALTH_TIMEOUT=5
TG_DEREG_DELAY=30

# 컨테이너/포트 (기본: 첫 컨테이너/첫 portMappings)
CONTAINER_INDEX="${CONTAINER_INDEX:-0}"
PORT_INDEX="${PORT_INDEX:-0}"

export AWS_DEFAULT_REGION="$REGION"

### =========================
### 인자 파싱
### =========================
NEW_TAG=""
RETAG_FROM=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)           NEW_TAG="$2"; shift 2 ;;
    --retag-from)    RETAG_FROM="$2"; shift 2 ;;
    --listener-arn)  ALB_LISTENER_ARN="$2"; shift 2 ;;
    --rule-arn)      RULE_ARN="$2"; shift 2 ;;
    --vpc-id)        VPC_ID="$2"; shift 2 ;;
    --tg-name)       TG_NAME="$2"; shift 2 ;;
    --health-path)   TG_HEALTH_PATH="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

command -v jq >/dev/null || { echo "ERROR: jq가 필요합니다."; exit 1; }
command -v aws >/dev/null || { echo "ERROR: aws CLI가 필요합니다."; exit 1; }

log(){ echo -e "[$(date +%H:%M:%S)] $*"; }
die(){ echo "ERROR: $*" >&2; exit 1; }

### =========================
### 0) 현재 서비스/TD 정보
### =========================
log "0) 현재 서비스가 사용하는 TaskDefinition 조회"
TD_OLD=$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" \
  --query 'services[0].taskDefinition' --output text)
[[ -n "${TD_OLD}" && "${TD_OLD}" != "None" ]] || die "서비스의 기존 TaskDefinition을 찾지 못함"

aws ecs describe-task-definition --task-definition "$TD_OLD" \
  --query 'taskDefinition' --output json >/tmp/td.json

CN=$(jq -r ".containerDefinitions[$CONTAINER_INDEX].name" /tmp/td.json)
PORT=$(jq -r ".containerDefinitions[$CONTAINER_INDEX].portMappings[$PORT_INDEX].containerPort // 8501" /tmp/td.json)
[[ "$CN" != "null" && "$CN" != "" ]] || die "컨테이너 이름을 찾지 못함 (index=$CONTAINER_INDEX)"
[[ "$PORT" != "null" && "$PORT" != "" ]] || die "컨테이너 포트를 찾지 못함 (index=$PORT_INDEX)"

OLD_IMAGE=$(jq -r ".containerDefinitions[$CONTAINER_INDEX].image" /tmp/td.json)
log "OLD TD : $TD_OLD"
log "CN/PORT: $CN / $PORT"
log "OLD IMG: $OLD_IMAGE"

OLD_TAG="${OLD_IMAGE##*:}"
[[ "$OLD_TAG" != "$OLD_IMAGE" ]] || OLD_TAG=""

### =========================
### 1) (선택) ECR 리태깅
### =========================
if [[ -n "$NEW_TAG" ]]; then
  SRC_TAG="${RETAG_FROM:-$OLD_TAG}"
  if [[ -n "$SRC_TAG" ]]; then
    log "1) ECR 리태깅: $SRC_TAG -> $NEW_TAG"
    MANIFEST=$(aws ecr batch-get-image --repository-name "$ECR_REPO_NAME" \
      --image-ids imageTag="$SRC_TAG" --query 'images[0].imageManifest' --output text)
    [[ -n "$MANIFEST" && "$MANIFEST" != "None" ]] || die "소스 태그($SRC_TAG)의 manifest를 찾지 못함"
    aws ecr put-image --repository-name "$ECR_REPO_NAME" \
      --image-tag "$NEW_TAG" --image-manifest "$MANIFEST" >/dev/null
  else
    log "WARN: 소스 태그가 없어 리태깅 생략"
  fi
fi

### =========================
### 2) 새 TD 등록(JSON 생성)
### =========================
if [[ -n "$NEW_TAG" ]]; then
  NEW_IMAGE="${IMAGE_BASE_URI}:${NEW_TAG}"
else
  NEW_IMAGE="$OLD_IMAGE"
fi
log "NEW IMG: $NEW_IMAGE"

jq --arg img "$NEW_IMAGE" --argjson ci "$CONTAINER_INDEX" '
  .containerDefinitions[$ci].image = $img
  | {
      family,
      taskRoleArn,
      executionRoleArn,
      networkMode,
      containerDefinitions,
      volumes,
      placementConstraints,
      requiresCompatibilities,
      cpu,
      memory
    }
  + (if .runtimePlatform   then {runtimePlatform:.runtimePlatform}     else {} end)
  + (if .ephemeralStorage  then {ephemeralStorage:.ephemeralStorage}   else {} end)
' /tmp/td.json >/tmp/td_reg.json

log "2) TD 새 리비전 등록"
TD_NEW=$(aws ecs register-task-definition --cli-input-json file:///tmp/td_reg.json \
  --query 'taskDefinition.taskDefinitionArn' --output text)
log "TD_NEW: $TD_NEW"

### =========================
### 3) Target Group 확보(기존 사용 or 생성)
### =========================
log "3) 서비스의 기존 TG 조회"
SERVICE_TG=$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" \
  --query 'services[0].loadBalancers[0].targetGroupArn' --output text 2>/dev/null || true)

if [[ -n "$SERVICE_TG" && "$SERVICE_TG" != "None" ]]; then
  TG_ARN="$SERVICE_TG"
  log "기존 서비스 TG 사용: $TG_ARN"
else
  # 기존 TG 없는 서비스라면 TG를 만들고, 서비스에 매핑할 예정
  [[ -n "$VPC_ID" ]] || die "VPC_ID가 필요합니다(새 TG 생성 케이스)"
  TG_NAME="${TG_NAME:-${SERVICE}-tg}"
  log "기존 TG 없음 → 새 TG 생성: $TG_NAME (VPC: $VPC_ID, target-type: $TG_TARGET_TYPE)"
  TG_ARN=$(aws elbv2 create-target-group \
    --name "$TG_NAME" \
    --protocol "$TG_PROTOCOL" \
    --port "$PORT" \
    --vpc-id "$VPC_ID" \
    --target-type "$TG_TARGET_TYPE" \
    --health-check-protocol "$TG_HEALTH_PROTOCOL" \
    --health-check-port "$TG_HEALTH_PORT" \
    --health-check-path "$TG_HEALTH_PATH" \
    --healthy-threshold-count $TG_HEALTHY_THRESHOLD \
    --unhealthy-threshold-count $TG_UNHEALTHY_THRESHOLD \
    --health-check-interval-seconds $TG_HEALTH_INTERVAL \
    --health-check-timeout-seconds $TG_HEALTH_TIMEOUT \
    --query 'TargetGroups[0].TargetGroupArn' --output text)
  log "TG 생성 완료: $TG_ARN"

  # 등록 지연 시간 설정(선택)
  aws elbv2 modify-target-group-attributes \
    --target-group-arn "$TG_ARN" \
    --attributes Key=deregistration_delay.timeout_seconds,Value=$TG_DEREG_DELAY >/dev/null
fi

### =========================
### 4) 서비스 업데이트(새 TD + TG 매핑)
### =========================
log "4) 서비스 롤링 배포 요청"
if [[ -n "${TG_ARN:-}" ]]; then
  aws ecs update-service \
    --cluster "$CLUSTER" --service "$SERVICE" \
    --task-definition "$TD_NEW" \
    --load-balancers "targetGroupArn=$TG_ARN,containerName=${CN},containerPort=${PORT}" \
    --health-check-grace-period-seconds 90 \
    --force-new-deployment >/dev/null
else
  aws ecs update-service \
    --cluster "$CLUSTER" --service "$SERVICE" \
    --task-definition "$TD_NEW" \
    --health-check-grace-period-seconds 90 \
    --force-new-deployment >/dev/null
fi
log "UpdateService OK → 서비스 안정화 대기(services-stable)"
aws ecs wait services-stable --cluster "$CLUSTER" --services "$SERVICE"
log "서비스 안정화 완료"

### =========================
### 5) TG 타겟 헬스 확인(최소 1 healthy 대기)
### =========================
if [[ -n "${TG_ARN:-}" ]]; then
  log "5) TargetGroup 타겟 헬스 대기 (healthy ≥ 1)"
  for i in {1..40}; do
    # RUNNING 태스크가 TG에 등록되고 헬스 체크 통과할 때까지 대기
    HEALTH_JSON=$(aws elbv2 describe-target-health --target-group-arn "$TG_ARN" || true)
    HEALTHY=$(echo "$HEALTH_JSON" | jq -r '[.TargetHealthDescriptions[]?.TargetHealth?.State=="healthy"] | length')
    TOTAL=$(echo "$HEALTH_JSON" | jq -r '.TargetHealthDescriptions | length')
    log "  - $HEALTHY / $TOTAL healthy"
    if [[ "${HEALTHY:-0}" -ge 1 ]]; then
      break
    fi
    sleep 5
  done
fi

### =========================
### 6) ALB 포워딩 갱신(규칙 or 기본동작)
### =========================
if [[ -n "${TG_ARN:-}" ]]; then
  if [[ -n "$RULE_ARN" ]]; then
    log "6) 지정 규칙 포워딩 갱신 → TG=$TG_ARN"
    aws elbv2 modify-rule --rule-arn "$RULE_ARN" \
      --actions Type=forward,TargetGroupArn="$TG_ARN" >/dev/null
  elif [[ -n "$ALB_LISTENER_ARN" ]]; then
    log "6) 리스너 기본동작 포워딩 갱신 → TG=$TG_ARN"
    aws elbv2 modify-listener --listener-arn "$ALB_LISTENER_ARN" \
      --default-actions Type=forward,TargetGroupArn="$TG_ARN" >/dev/null
  else
    log "6) ALB 갱신 생략(리스너/규칙 ARN 미지정)"
  fi
else
  log "6) ALB 갱신 생략(TG 미지정)"
fi

### =========================
### 7) 요약/검증 힌트
### =========================
log "7) 현재 서비스 요약"
aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" \
  --query 'services[0].{running:runningCount,pending:pendingCount,td:taskDefinition,lb:loadBalancers}' --output json

if [[ -n "${TG_ARN:-}" ]]; then
  log "TargetGroup 상태:"
  aws elbv2 describe-target-health --target-group-arn "$TG_ARN" \
    --query 'TargetHealthDescriptions[].{Id:Target.Id,Port:Target.Port,State:TargetHealth.State,Reason:TargetHealth.Reason}' --output json
fi

echo
echo "검증 Tip:"
echo "  aws ecs list-tasks --cluster $CLUSTER --service-name $SERVICE --desired-status RUNNING --output text"
echo "  aws ecs describe-tasks --cluster $CLUSTER --tasks <TASK_ARN> --query 'tasks[0].containers[].{name:name,image:image,imageDigest:imageDigest}' --output json"
echo "  aws elbv2 describe-rules --listener-arn $ALB_LISTENER_ARN --query 'Rules[].{Rule:RuleArn,Action:Actions}' --output json"
