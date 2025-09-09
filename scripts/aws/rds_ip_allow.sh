#!/usr/bin/env bash
set -euo pipefail

REGION=${REGION:-ap-northeast-2}
DB_ID=${DB_ID:-h2o2-db-prd}   
# DB_SG_ID 를 알고 있으면 주고, 모르면 비워두세요: 스크립트가 DB에서 자동 추출
DB_SG_ID=${DB_SG_ID:-sg-097b5ebe2a255674a}

cmd=${1:-status}   # 사용법: ./rds-ip-allow.sh open|close|status

get_db_sg_id() {
  if [[ -n "${DB_SG_ID:-}" ]]; then
    echo "$DB_SG_ID"; return
  fi
  aws rds describe-db-instances --region "$REGION" --db-instance-identifier "$DB_ID" \
    --query 'DBInstances[0].VpcSecurityGroups[0].VpcSecurityGroupId' --output text
}

get_db_endpoint() {
  aws rds describe-db-instances --region "$REGION" --db-instance-identifier "$DB_ID" \
    --query 'DBInstances[0].Endpoint.Address' --output text
}

case "$cmd" in
  open)
    MYIP="$(curl -s https://checkip.amazonaws.com)/32"
    SG=$(get_db_sg_id)
    echo "Allowing $MYIP to TCP/5432 on SG $SG"
    aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG" \
      --protocol tcp --port 5432 --cidr "$MYIP" >/dev/null 2>&1 || true
    echo "Done. Endpoint: $(get_db_endpoint):5432"
    ;;

  close)
    MYIP="$(curl -s https://checkip.amazonaws.com)/32"
    SG=$(get_db_sg_id)
    echo "Revoking $MYIP from TCP/5432 on SG $SG"
    aws ec2 revoke-security-group-ingress --region "$REGION" --group-id "$SG" \
      --protocol tcp --port 5432 --cidr "$MYIP" >/dev/null 2>&1 || true
    echo "Done."
    ;;

  status)
    SG=$(get_db_sg_id)
    echo "DB_SG_ID=$SG"
    echo "Inbound 5432 rules:"
    aws ec2 describe-security-groups --region "$REGION" --group-ids "$SG" \
      --query 'SecurityGroups[0].IpPermissions[?FromPort==`5432` && ToPort==`5432`]' --output json
    echo "Endpoint: $(get_db_endpoint):5432"
    ;;

  *)
    echo "Usage: $0 {open|close|status}"
    exit 1
    ;;
esac

