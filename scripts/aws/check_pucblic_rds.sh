REGION=ap-northeast-2
DB_ID=h2o2-db-prd

# 현재 상태 확인
aws rds describe-db-instances --region $REGION --db-instance-identifier $DB_ID \
  --query 'DBInstances[0].{Endpoint:Endpoint.Address,Public:PubliclyAccessible,SubnetGroup:DBSubnetGroup.DBSubnetGroupName}' --output table

# 필요시 퍼블릭 접근 켜기 (DB Subnet Group이 "퍼블릭 서브넷"으로 구성돼 있어야 함)
aws rds modify-db-instance --region $REGION --db-instance-identifier $DB_ID \
  --publicly-accessible --apply-immediately
