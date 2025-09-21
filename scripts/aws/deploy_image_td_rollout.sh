#!/usr/bin/env bash
set -euo pipefail

### sample command
### ./deploy.sh --tag prd --no-retag --health-path /v1/health


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

# ALB / TG (옵션)
ALB_LISTENER_ARN=""   # 예: arn:aws:elasticloadbalancing:...:listener/app/xxx/yyy/zzz
RULE_ARN=""           # 특정 규칙 갱신 시 지정
VPC_ID=""             # 새 TG 생성 시 필요(기존 TG를 쓸 경우 생략 가능)

# 헬스체크/포트/TG 설정(필요 시 수정)
TG_NAME=""
TG_TARGET_TYPE="ip"   # Fargate는 ip 권장
TG_PROTOCOL="HTTP"
TG_HEALTH_PROTOCOL="HTTP"
TG_HEALTH_PATH="/"    # FastAPI면 /v1/health 권장
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
NO_RETAG=false
PIN_BY_DIGEST=false

usage(){
  cat <<EOF
Usage: $0 [options]
  --tag <NEW_TAG>          서비스에 적용할 새 이미지 태그(예: prd-20250916-1)
  --no-retag               ECR retag(put-image) 건너뛰고 NEW_TAG를 직접 사용
  --pin-by-digest          NEW_TAG를 digest로 해석하여 TD에 IMAGE@sha256 형식으로 고정
  --retag-from <TAG>       (선택) retag 소스 태그
  --listener-arn <ARN>     ALB 리스너 ARN
  --rule-arn <ARN>         수정할 리스너 규칙 ARN
  --vpc-id <VPCID>         새 TG 생성 시 필요
  --tg-name <NAME>         새 TG 이름(미지정 시 SERVICE-tg)
  --health-path <PATH>     헬스체크 경로(기본 /)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)           NEW_TAG="$2"; shift 2 ;;
    --retag-from)    RETAG_FROM="$2"; shift 2 ;;
    --no-retag)      NO_RETAG=true; shift 1 ;;
    --pin-by-digest) PIN_BY_DIGEST=true; shift 1 ;;
    --listener-arn)  ALB_LISTENER_ARN="$2"; shift 2 ;;
    --rule-arn)      RULE_ARN="$2"; shift 2 ;;
    --vpc-id)        VPC_ID="$2"; shift 2 ;;
    --tg-name)       TG_NAME="$2"; shift 2 ;;
    --health-path)   TG_HEALTH_PATH="$2"; shift 2 ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

command -v jq  >/dev/null || { echo "ERROR: jq가 필요합니다."; exit 1; }
command -v aws >/dev/null || { echo "ERROR: aws CLI가 필요합니다."; exit 1; }

log(){ echo -e "[$(date +%H:%M:%S)] $*"; }
die(){ echo "ERROR: $*" >&2; exit 1; }
cleanup(){ rm -f /tmp/td.json /tmp/td_reg.json; }
trap cleanup EXIT

# ===== helpers =====
get_running_task_digest() {
  local arn
  arn=$(aws ecs list-tasks \
    --cluster "$CLUSTER" --service-name "$SERVICE" --desired-status RUNNING \
    --query 'taskArns[0]' --output text 2>/dev/null | tr -d '\r')
  [[ -n "$arn" && "$arn" != "None" ]] || return 1
  aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$arn" \
    --query "tasks[0].containers[$CONTAINER_INDEX].imageDigest" \
    --output text 2>/dev/null | tr -d '\r'
}

get_latest_ecr_digest() {
  aws ecr describe-images --repository-name "$ECR_REPO_NAME" \
    --query 'reverse(sort_by(imageDetails,&imagePushedAt))[0].imageDigest' \
    --output text 2>/dev/null | tr -d '\r'
}

get_digest_by_tag() {
  local tag="$1"
  aws ecr describe-images --repository-name "$ECR_REPO_NAME" \
    --image-ids imageTag="$tag" \
    --query 'imageDetails[0].imageDigest' \
    --output text 2>/dev/null | tr -d '\r'
}

ensure_tag_exists() {
  local tag="$1"
  local digest
  digest=$(get_digest_by_tag "$tag" || true)
  [[ -n "$digest" && "$digest" != "None" ]] || die "ECR에 태그가 존재하지 않음: $ECR_REPO_NAME:$tag"
}

### =========================
### 0) 현재 서비스/TD 정보
### =========================
log "0) 현재 서비스가 사용하는 TaskDefinition 조회"
TD_OLD=$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" \
  --query 'services[0].taskDefinition' --output text | tr -d '\r')
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

### =========================
### 1) (선택) ECR 리태깅 — 혹은 완전 스킵
### =========================
MANIFEST=""
TARGET_DIGEST=""

if [[ -n "$NEW_TAG" ]]; then
  if $NO_RETAG; then
    log "1) --no-retag 지정: ECR put-image 생략, NEW_TAG 존재 여부만 확인"
    ensure_tag_exists "$NEW_TAG"
  else
    log "1) ECR 리태깅 준비(소스 manifest 확보)"
    # (선택) 사용자가 --retag-from <tag>를 지정한 경우 우선 사용
    if [[ -n "${RETAG_FROM:-}" ]]; then
      log " - 태그 기반 시도: $RETAG_FROM"
      MANIFEST=$(aws ecr batch-get-image --repository-name "$ECR_REPO_NAME" \
        --image-ids imageTag="$RETAG_FROM" \
        --accepted-media-types application/vnd.docker.distribution.manifest.list.v2+json \
        --accepted-media-types application/vnd.docker.distribution.manifest.v2+json \
        --query 'images[0].imageManifest' --output text 2>/dev/null | tr -d '\r' || true)
      [[ -n "$MANIFEST" && "$MANIFEST" != "None" ]] && log " - 태그에서 manifest 확보" || MANIFEST=""
    fi

    # digest 기반: 실행 중 태스크 → 없으면 ECR 최신
    if [[ -z "$MANIFEST" ]]; then
      TARGET_DIGEST=$(get_running_task_digest || true)
      if [[ -n "$TARGET_DIGEST" && "$TARGET_DIGEST" != "None" ]]; then
        log " - 실행중 태스크 digest: $TARGET_DIGEST"
      else
        TARGET_DIGEST=$(get_latest_ecr_digest || true)
        [[ -n "$TARGET_DIGEST" && "$TARGET_DIGEST" != "None" ]] || die "소스 digest를 찾지 못함 (RUNNING 태스크/리포지토리 최신 이미지 없음)"
        log " - ECR 최신 digest: $TARGET_DIGEST"
      fi

      MANIFEST=$(aws ecr batch-get-image --repository-name "$ECR_REPO_NAME" \
        --image-ids imageDigest="$TARGET_DIGEST" \
        --accepted-media-types application/vnd.docker.distribution.manifest.list.v2+json \
        --accepted-media-types application/vnd.docker.distribution.manifest.v2+json \
        --query 'images[0].imageManifest' --output text 2>/dev/null | tr -d '\r' || true)
      [[ -n "$MANIFEST" && "$MANIFEST" != "None" ]] || die "소스 digest의 manifest를 찾지 못함"
    fi

    log " - put-image → tag: $NEW_TAG"
    aws ecr put-image --repository-name "$ECR_REPO_NAME" \
      --image-tag "$NEW_TAG" --image-manifest "$MANIFEST" >/dev/null
  fi
fi

### =========================
### 2) 새 TD 등록(JSON 생성) — 태그/다이제스트 고정 선택
### =========================
if [[ -n "$NEW_TAG" ]]; then
  if $PIN_BY_DIGEST; then
    DIG=$(get_digest_by_tag "$NEW_TAG" || true)
    [[ -n "$DIG" && "$DIG" != "None" ]] || die "NEW_TAG → digest 해석 실패: $NEW_TAG"
    NEW_IMAGE="${IMAGE_BASE_URI}@${DIG}"   # ★ digest 고정
    log "NEW IMG (pinned digest): $NEW_IMAGE  (tag=$NEW_TAG, digest=$DIG)"
  else
    NEW_IMAGE="${IMAGE_BASE_URI}:${NEW_TAG}"  # 태그 사용
    log "NEW IMG (by tag): $NEW_IMAGE"
  fi
else
  NEW_IMAGE="$OLD_IMAGE"
  log "NEW IMG: (변경 없음) $NEW_IMAGE"
fi

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
  --query 'taskDefinition.taskDefinitionArn' --output text | tr -d '\r')
log "TD_NEW: $TD_NEW"

### =========================
### 3) Target Group 확보(기존 사용 or 생성)
### =========================
log "3) 서비스의 기존 TG 조회"
SERVICE_TG=$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" \
  --query 'services[0].loadBalancers[0].targetGroupArn' --output text 2>/dev/null | tr -d '\r' || true)

if [[ -n "$SERVICE_TG" && "$SERVICE_TG" != "None" ]]; then
  TG_ARN="$SERVICE_TG"
  log " - 기존 서비스 TG 사용: $TG_ARN"
else
  [[ -n "$VPC_ID" ]] || die "VPC_ID가 필요합니다(새 TG 생성 케이스)"
  TG_NAME="${TG_NAME:-${SERVICE}-tg}"
  log " - 기존 TG 없음 → 새 TG 생성: $TG_NAME (VPC: $VPC_ID, target-type: $TG_TARGET_TYPE)"
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
    --query 'TargetGroups[0].TargetGroupArn' --output text | tr -d '\r')
  log " - TG 생성 완료: $TG_ARN"

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
### 5) TG 타깃 헬스 확인(최소 1 healthy 대기)
### =========================
if [[ -n "${TG_ARN:-}" ]]; then
  log "5) TargetGroup 타깃 헬스 대기 (healthy ≥ 1)"
  for i in {1..40}; do
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
