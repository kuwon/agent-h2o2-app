#!/usr/bin/env bash
set -Eeuo pipefail

REGION=ap-northeast-2

# 확인할 서브넷들 (원하는 값으로 교체)
SUBNETS=("subnet-08e75869bc55b5df1" "subnet-095f958fbf81ed9d4" "subnet-0f55d4128b7e52e0a" "subnet-08d15ceb04b65afdd")

# 반드시 초기화 (nounset 안전)
PUBLIC=()
PRIVATE=()

for s in "${SUBNETS[@]}"; do
  # 서브넷의 VPC / AZ / CIDR
  OUT=$(aws ec2 describe-subnets --region "$REGION" --subnet-ids "$s" \
        --query 'Subnets[0].[VpcId,AvailabilityZone,CidrBlock]' --output text)
  VPC_ID=$(echo "$OUT" | awk '{print $1}')
  AZ=$(echo "$OUT"     | awk '{print $2}')
  CIDR=$(echo "$OUT"   | awk '{print $3}')

  # 1) 서브넷에 직접 연결된 라우트테이블
  RT=$(aws ec2 describe-route-tables --region "$REGION" \
        --filters "Name=association.subnet-id,Values=$s" \
        --query 'RouteTables[0].RouteTableId' --output text)

  # 1-1) 없으면 VPC의 main 라우트테이블 사용
  if [[ "$RT" == "None" || -z "$RT" ]]; then
    RT=$(aws ec2 describe-route-tables --region "$REGION" \
          --filters "Name=vpc-id,Values=$VPC_ID" "Name=association.main,Values=true" \
          --query 'RouteTables[0].RouteTableId' --output text)
  fi

  # 2) 기본 라우트(0.0.0.0/0)의 대상 확인
  ROUTE_OUT=$(aws ec2 describe-route-tables --region "$REGION" --route-table-ids "$RT" \
               --query 'RouteTables[0].Routes[?DestinationCidrBlock==`0.0.0.0/0`].[GatewayId,NatGatewayId]' \
               --output text)
  GATEWAY=$(echo "$ROUTE_OUT" | awk '{print $1}')
  NATGW=$(echo   "$ROUTE_OUT" | awk '{print $2}')

  CLASS=""
  if [[ "$GATEWAY" == igw-* ]]; then
    CLASS="PUBLIC"
  elif [[ "$NATGW" == nat-* ]]; then
    CLASS="PRIVATE"
  else
    # 3) 보조판별: 퍼블릭 IP 자동할당 여부 (단수형 API!)
    MAP=$(aws ec2 describe-subnet-attribute --region "$REGION" \
           --subnet-id "$s" --attribute mapPublicIpOnLaunch \
           --query 'MapPublicIpOnLaunch.Value' --output text)
    if [[ "$MAP" == "True" ]]; then CLASS="PUBLIC"; else CLASS="PRIVATE"; fi
  fi

  if [[ "$CLASS" == "PUBLIC" ]]; then PUBLIC+=("$s"); else PRIVATE+=("$s"); fi
  printf "%-14s  %-20s  %-8s  %s\n" "$AZ" "$s" "$CLASS" "$CIDR"
done

echo
echo "PUBLIC  : ${PUBLIC[*]:-}"
echo "PRIVATE : ${PRIVATE[*]:-}"

# 멀티 AZ에서 2개 고르는 함수 (빈 입력 안전)
pick_two_by_az () {
  local arr=("$@")
  if (( ${#arr[@]} == 0 )); then
    echo ""   # 빈 문자열 반환
    return 0
  fi
  for s in "${arr[@]}"; do
    AZ=$(aws ec2 describe-subnets --region "$REGION" --subnet-ids "$s" \
         --query 'Subnets[0].AvailabilityZone' --output text)
    echo "$AZ $s"
  done | sort | awk '!seen[$1]++ {print $2}' | head -n 2 | paste -sd" " -
}

SUBNETS_PUBLIC=""
SUBNETS_TASK=""

# 배열이 비어있을 때는 호출하지 않음 (nounset 안전)
if (( ${#PUBLIC[@]}  > 0 )); then SUBNETS_PUBLIC="$(pick_two_by_az "${PUBLIC[@]}")"; fi
if (( ${#PRIVATE[@]} > 0 )); then SUBNETS_TASK="$(pick_two_by_az "${PRIVATE[@]}")"; fi

echo "SUBNETS_PUBLIC=\"${SUBNETS_PUBLIC}\""
echo "SUBNETS_TASK=\"${SUBNETS_TASK}\""

