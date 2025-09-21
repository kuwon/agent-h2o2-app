#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# add_execution_to_secret.sh (완성본)
# - Secrets Manager의 JSON 시크릿에서 특정 키들을 컨테이너 env로 주입
# - Task Role(ecs exec 필수) 자동 보강
# - 새 Task Definition 등록 후 서비스 롤링 배포
# ------------------------------------------------------------------
# 요구 사항:
#   - awscli, jq 설치
#   - 실행 주체의 IAM 권한: ecs, iam(역할 생성/정책 부여 시), secretsmanager, sts, elbv2(없어도 됨)
#
# 사용 예:
#   bash ./scripts/aws/add_execution_to_secret.sh \
#     --region ap-northeast-2 \
#     --cluster h2o2-cluster \
#     --service h2o2-service \
#     --family h2o2-task \
#     --secret-arn arn:aws:secretsmanager:ap-northeast-2:123456789012:secret:h2o2-db-secrets-XXXX \
#     --openai-key OPENAI_API_KEY \
#     --add-keys DATABASE_URL,DB_HOST,DB_PORT,DB_USER,DB_PASS,DB_NAME \
#     --enable-exec true
#
# 참고:
#   - 시크릿은 JSON 구조여야 하며, 각 키를 valueFrom에  :KEY::  suffix로 지정합니다.
#   - Execution Role은 secretsmanager:GetSecretValue 권한 필요
#   - Task Role은 ECS Exec용 ssmmessages 권한 필요
# ------------------------------------------------------------------

# ===== 기본값 =====
REGION="ap-northeast-2"
CLUSTER=""
SERVICE=""
FAMILY=""

SECRET_ARN=""                 # Secrets Manager Secret ARN
CONTAINER_INDEX="${CONTAINER_INDEX:-0}"

OPENAI_KEY_NAME="OPENAI_API_KEY"
ADD_KEYS="DATABASE_URL,DB_HOST,DB_PORT,DB_USER,DB_PASS,DB_NAME"  # 콤마구분 목록
ENABLE_EXEC="false"             # 서비스에서 ECS Exec 활성화 여부

TASK_ROLE_NAME="h2o2-ecs-task-role"
EXEC_ROLE_NAME="ecsTaskExecutionRole"   # 일반적으로 기본 실행 역할 이름

log(){ echo -e "[$(date +%H:%M:%S)] $*"; }
die(){ echo "ERROR: $*" >&2; exit 1; }

need_bin(){
  command -v "$1" >/dev/null || die "필요한 바이너리($1)가 없습니다."
}

usage(){
  cat <<EOF
사용법:
  $0 --cluster <CLUSTER> --service <SERVICE> --secret-arn <SECRET_ARN> [옵션]

필수:
  --cluster           ECS Cluster 이름
  --service           ECS Service 이름
  --secret-arn        Secrets Manager의 Secret ARN (JSON 시크릿)

옵션:
  --region            AWS 리전 (기본: ${REGION})
  --family            Task Definition Family (미지정 시 기존 TD의 family 사용)
  --container-index   시크릿을 주입할 컨테이너 인덱스 (기본: ${CONTAINER_INDEX})
  --openai-key        OPENAI 키의 env 이름 (기본: ${OPENAI_KEY_NAME})
  --add-keys          추가 시크릿 키 콤마 목록 (기본: ${ADD_KEYS})
  --enable-exec       서비스에 ECS Exec 활성화 true/false (기본: ${ENABLE_EXEC})
  --task-role-name    새로 만들/사용할 Task Role 이름 (기본: ${TASK_ROLE_NAME})
  --exec-role-name    Execution Role 이름 (기본: ${EXEC_ROLE_NAME})

예시:
  $0 --cluster h2o2-cluster --service h2o2-service \
     --secret-arn arn:aws:secretsmanager:ap-northeast-2:123:secret:h2o2-db-secrets-XXX \
     --openai-key OPENAI_API_KEY \
     --add-keys DATABASE_URL,DB_HOST,DB_PORT,DB_USER,DB_PASS,DB_NAME \
     --enable-exec true
EOF
}

# ===== 인자 파싱 =====
while [[ $# -gt 0 ]]; do
  case "$1" in
    --region) REGION="$2"; shift 2 ;;
    --cluster) CLUSTER="$2"; shift 2 ;;
    --service) SERVICE="$2"; shift 2 ;;
    --family) FAMILY="$2"; shift 2 ;;
    --secret-arn) SECRET_ARN="$2"; shift 2 ;;
    --container-index) CONTAINER_INDEX="$2"; shift 2 ;;
    --openai-key) OPENAI_KEY_NAME="$2"; shift 2 ;;
    --add-keys) ADD_KEYS="$2"; shift 2 ;;
    --enable-exec) ENABLE_EXEC="$2"; shift 2 ;;
    --task-role-name) TASK_ROLE_NAME="$2"; shift 2 ;;
    --exec-role-name) EXEC_ROLE_NAME="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "알 수 없는 인자: $1" ;;
  esac
done

need_bin aws
need_bin jq

[[ -n "$CLUSTER" && -n "$SERVICE" && -n "$SECRET_ARN" ]] || { usage; die "필수 인자 누락"; }

export AWS_DEFAULT_REGION="$REGION"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
[[ -n "$ACCOUNT_ID" && "$ACCOUNT_ID" != "None" ]] || die "Account ID 조회 실패"

# ===== 현재 서비스/TD 정보 =====
log "현재 서비스가 사용하는 TaskDefinition 조회"
TD_OLD="$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" \
  --query 'services[0].taskDefinition' --output text)"
[[ -n "$TD_OLD" && "$TD_OLD" != "None" ]] || die "기존 TaskDefinition을 찾지 못함"

aws ecs describe-task-definition --task-definition "$TD_OLD" \
  --query 'taskDefinition' --output json >/tmp/td.json

OLD_FAMILY="$(jq -r '.family' /tmp/td.json)"
CN="$(jq -r ".containerDefinitions[$CONTAINER_INDEX].name" /tmp/td.json)"
[[ "$CN" != "null" && -n "$CN" ]] || die "컨테이너 이름을 찾지 못함 (index=$CONTAINER_INDEX)"

TASK_ROLE_OLD="$(jq -r '.taskRoleArn // empty' /tmp/td.json)"
EXEC_ROLE_OLD="$(jq -r '.executionRoleArn // empty' /tmp/td.json)"
[[ -z "$FAMILY" ]] && FAMILY="$OLD_FAMILY"

log "TD_OLD     : $TD_OLD"
log "FAMILY     : $FAMILY"
log "CONTAINER  : $CN"
log "TASK ROLE  : ${TASK_ROLE_OLD:-<none>}"
log "EXEC ROLE  : ${EXEC_ROLE_OLD:-<none>}"

# ===== Task Role 확보(없으면 생성 + 정책 부여) =====
TASK_ROLE_ARN="${TASK_ROLE_OLD:-arn:aws:iam::${ACCOUNT_ID}:role/${TASK_ROLE_NAME}}"

if [[ -z "$TASK_ROLE_OLD" ]]; then
  if ! aws iam get-role --role-name "$TASK_ROLE_NAME" >/dev/null 2>&1; then
    log "Task Role 미존재 → 생성: $TASK_ROLE_NAME"
    aws iam create-role \
      --role-name "$TASK_ROLE_NAME" \
      --assume-role-policy-document '{
        "Version":"2012-10-17",
        "Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]
      }' >/dev/null
  else
    log "Task Role 존재: $TASK_ROLE_NAME"
  fi

  # ECS Exec 통신 권한(SSM Messages)
  log "Task Role에 ECS Exec 권한(SSM Messages) 부여/업데이트"
  aws iam put-role-policy \
    --role-name "$TASK_ROLE_NAME" \
    --policy-name EcsExecSsmMessages \
    --policy-document '{
      "Version":"2012-10-17",
      "Statement":[{"Effect":"Allow","Action":[
        "ssmmessages:CreateControlChannel",
        "ssmmessages:CreateDataChannel",
        "ssmmessages:OpenControlChannel",
        "ssmmessages:OpenDataChannel"
      ],"Resource":"*"}]
    }' >/dev/null
else
  log "기존 TD의 Task Role 재사용: $TASK_ROLE_OLD"
  TASK_ROLE_ARN="$TASK_ROLE_OLD"
fi

# ===== Execution Role 확보(없으면 기본명 사용) =====
EXEC_ROLE_ARN="${EXEC_ROLE_OLD:-arn:aws:iam::${ACCOUNT_ID}:role/${EXEC_ROLE_NAME}}"
if [[ -z "$EXEC_ROLE_OLD" ]]; then
  log "Execution Role 미지정 → 지정: $EXEC_ROLE_NAME (ARN: $EXEC_ROLE_ARN)"
  # 필요시 여기서 secretsmanager:GetSecretValue 최소권한을 붙이도록 안내만 출력
  log "참고: Execution Role에 secretsmanager:GetSecretValue 권한이 필요합니다."
fi

# ===== 시크릿 키 목록 구성 =====
# - OPENAI 키 포함
IFS=',' read -r -a EXTRA_KEYS <<< "$ADD_KEYS"
SECRET_KEYS=("${EXTRA_KEYS[@]}")
# OPENAI 키가 중복으로 안 들어가도록 검사
already=false
for k in "${SECRET_KEYS[@]}"; do
  [[ "$k" == "$OPENAI_KEY_NAME" ]] && already=true && break
done
[[ "$already" == "false" ]] && SECRET_KEYS+=("$OPENAI_KEY_NAME")

log "주입할 시크릿 키: ${SECRET_KEYS[*]}"

# ===== 시크릿 JSON 유효성 점검(경고만) =====
if SECRET_STR="$(aws secretsmanager get-secret-value --secret-id "$SECRET_ARN" --query SecretString --output text 2>/dev/null)"; then
  if ! echo "$SECRET_STR" | jq -e . >/dev/null 2>&1; then
    log "WARN: 이 시크릿은 JSON이 아닙니다. KEY suffix(:KEY::) 방식은 동작하지 않을 수 있습니다."
  else
    for k in "${SECRET_KEYS[@]}"; do
      if ! echo "$SECRET_STR" | jq -e --arg k "$k" 'has($k)' >/dev/null; then
        log "WARN: 시크릿에 키 '$k' 가 없습니다. (추가를 권장)"
      fi
    done
  fi
else
  log "WARN: SecretString 조회 실패. 권한/ARN을 확인하세요."
fi

# ===== 컨테이너 정의에 secrets upsert =====
# 새로운 secrets JSON 배열 생성
NEW_SECRETS_JSON="["
for idx in "${!SECRET_KEYS[@]}"; do
  key="${SECRET_KEYS[$idx]}"
  # Secrets Manager JSON 키 참조는 :KEY:: suffix를 사용
  # 예: arn:...:secret:my-secret:OPENAI_API_KEY::
  valueFrom="${SECRET_ARN}:${key}::"
  entry=$(jq -nc --arg name "$key" --arg vf "$valueFrom" '{name:$name, valueFrom:$vf}')
  if [[ $idx -gt 0 ]]; then NEW_SECRETS_JSON+=", "; fi
  NEW_SECRETS_JSON+="$entry"
done
NEW_SECRETS_JSON+="]"

log "컨테이너($CONTAINER_INDEX:$CN)에 secrets upsert 적용"
jq --argjson ci "$CONTAINER_INDEX" --argjson news "$NEW_SECRETS_JSON" '
  .containerDefinitions[$ci].secrets =
    (((.containerDefinitions[$ci].secrets // []) | map(select(.name as $n | ($news | map(.name) | index($n)) | not)))
     + $news)
' /tmp/td.json >/tmp/td_secrets.json
mv /tmp/td_secrets.json /tmp/td.json

# ===== 새 TD 등록(JSON 축약 구성) =====
jq '
  {
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
  + (if .runtimePlatform  then {runtimePlatform:.runtimePlatform}  else {} end)
  + (if .ephemeralStorage then {ephemeralStorage:.ephemeralStorage} else {} end)
' /tmp/td.json >/tmp/td_reg_base.json

# taskRoleArn / executionRoleArn 보강 주입
jq --arg taskRoleArn "$TASK_ROLE_ARN" --arg execRoleArn "$EXEC_ROLE_ARN" '
  .taskRoleArn = $taskRoleArn
  | .executionRoleArn = $execRoleArn
' /tmp/td_reg_base.json >/tmp/td_reg.json

log "새 Task Definition 등록"
TD_NEW="$(aws ecs register-task-definition \
  --cli-input-json file:///tmp/td_reg.json \
  --query 'taskDefinition.taskDefinitionArn' --output text)"
log "TD_NEW: $TD_NEW"

# ===== 서비스 업데이트 (롤링) =====
log "서비스 업데이트 (새 TD 적용)"
if [[ "$ENABLE_EXEC" == "true" ]]; then
  aws ecs update-service \
    --cluster "$CLUSTER" --service "$SERVICE" \
    --task-definition "$TD_NEW" \
    --enable-execute-command \
    --force-new-deployment >/dev/null
else
  aws ecs update-service \
    --cluster "$CLUSTER" --service "$SERVICE" \
    --task-definition "$TD_NEW" \
    --force-new-deployment >/dev/null
fi

log "서비스 안정화 대기"
aws ecs wait services-stable --cluster "$CLUSTER" --services "$SERVICE"

# ===== 요약 출력 =====
log "적용 요약"
aws ecs describe-task-definition --task-definition "$TD_NEW" \
  --query 'taskDefinition.{family:family,taskRoleArn:taskRoleArn,executionRoleArn:executionRoleArn,container:containerDefinitions[0].{name:name,secrets:secrets}}'

echo
log "검증 팁:"
echo "  aws ecs list-tasks --cluster $CLUSTER --service-name $SERVICE --desired-status RUNNING --region $REGION"
echo "  aws ecs execute-command --cluster $CLUSTER --task <TASK_ARN> --container $CN --command \"env | grep -E 'OPENAI_API_KEY'\" --interactive --region $REGION"

