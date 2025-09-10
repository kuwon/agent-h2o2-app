# workspace/prd_resources.py
from os import getenv
from agno.aws.app.streamlit import Streamlit
from agno.aws.resource.ec2 import InboundRule, SecurityGroup
from agno.aws.resource.ecs import EcsCluster
from agno.aws.resource.rds import DbInstance, DbSubnetGroup
from agno.aws.resource.reference import AwsReference
from agno.aws.resource.s3 import S3Bucket
from agno.aws.resource.secret import SecretsManager
from agno.aws.resources import AwsResources
from agno.docker.resource.image import DockerImage
from agno.docker.resources import DockerResources
from workspace.settings import ws_settings

from agno.aws.resource.ecs import EcsTaskDefinition

# -----------------------------------------------------------------------------
# Global toggles
skip_delete: bool = False
save_output: bool = True

# -----------------------------------------------------------------------------
# Constants derived from what you've shared
SUBNETS_PUBLIC = [
    "subnet-095f958fbf81ed9d4",
    "subnet-08e75869bc55b5df1",
]
SUBNETS_ALL = ws_settings.subnet_ids  # ["subnet-08e7...", "subnet-095f...", "subnet-0f55...", "subnet-08d1..."]

# -----------------------------------------------------------------------------
# Image (ECR)
prd_image = DockerImage(
    name=f"{ws_settings.image_repo}/{ws_settings.image_name}",
    tag=ws_settings.prd_env,                  # :prd
    enabled=ws_settings.build_images,         # True
    path=str(ws_settings.ws_root),
    platforms=["linux/amd64", "linux/arm64"],
    push_image=ws_settings.push_images,       # True
)

# -----------------------------------------------------------------------------
# (Optional) Storage & Secrets
prd_bucket = S3Bucket(
    name="h2o2-storage",
    enabled=False,  # 필요해지면 True
    acl="private",
    skip_delete=skip_delete,
    save_output=save_output,
)

prd_secret = SecretsManager(
    name="h2o2-secrets",
    group="app",
    secret_files=[ws_settings.ws_root.joinpath("workspace/secrets/prd_app_secrets.yml")],
    skip_delete=skip_delete,
    save_output=save_output,
)

prd_db_secret = SecretsManager(
    name="h2o2-db-secrets",
    group="db",
    secret_files=[ws_settings.ws_root.joinpath("workspace/secrets/prd_db_secrets.yml")],
    skip_delete=skip_delete,
    save_output=save_output,
)

# -----------------------------------------------------------------------------
# Security Groups
prd_lb_sg = SecurityGroup(
    name="h2o2-alb-sg",
    group="app",
    description="ALB SG (internet-facing 80/443)",
    inbound_rules=[
        InboundRule(description="HTTP from internet", port=80, cidr_ip="0.0.0.0/0"),
        InboundRule(description="HTTPS from internet", port=443, cidr_ip="0.0.0.0/0"),
    ],
    subnets=SUBNETS_ALL,
    skip_delete=skip_delete,
    save_output=save_output,
)

prd_svc_sg = SecurityGroup(
    name="h2o2-svc-sg",
    group="app",
    description="Service SG (8501 from ALB)",
    inbound_rules=[
        InboundRule(description="ALB -> 8501", port=8501, security_group_id=AwsReference(prd_lb_sg.get_security_group_id)),
        # 필요시 FastAPI 8000도 오픈
        # InboundRule(description="ALB -> 8000", port=8000, security_group_id=AwsReference(prd_lb_sg.get_security_group_id)),
    ],
    depends_on=[prd_lb_sg],
    subnets=SUBNETS_ALL,
    skip_delete=skip_delete,
    save_output=save_output,
)

# DB는 앱 SG만 허용
prd_db_sg = SecurityGroup(
    name="h2o2-db-sg",
    group="db",
    description="RDS PostgreSQL SG (5432 from service)",
    inbound_rules=[
        InboundRule(description="Service -> DB 5432", port=5432, security_group_id=AwsReference(prd_svc_sg.get_security_group_id)),
    ],
    depends_on=[prd_svc_sg],
    subnets=SUBNETS_ALL,
    skip_delete=skip_delete,
    save_output=save_output,
)

# # -----------------------------------------------------------------------------
# # Task Definition
# prd_td = EcsTaskDefinition(
#     name="h2o2-td",            # Agno 리소스명(ag ws patch --name td 가 찾는 대상)
#     family="h2o2-task",        # AWS 상의 TD family (이미 존재하는 h2o2-task 와 정렬)
#     requires_compatibilities=["FARGATE"],
#     network_mode="awsvpc",
#     cpu="1024",
#     memory="2048",
#     execution_role_arn=f"arn:aws:iam::{getenv('AWS_ACCOUNT_ID','')}:role/ecsTaskExecutionRole",  # 없으면 콘솔/CLI로 먼저 생성
#     # 컨테이너 정의
#     containers=[{
#         "name": "api",  # TD의 containerDefinitions[0].name 과 서비스 loadBalancers.containerName이 일치해야 함
#         "image": f"{ws_settings.image_repo}/{ws_settings.image_name}:{ws_settings.prd_env}",
#         "command": ["/app/scripts/entrypoint.sh","serve"],
#         "portMappings": [{"containerPort": 8501, "protocol": "tcp"}],
#         "logConfiguration": {
#             "logDriver": "awslogs",
#             "options": {
#                 "awslogs-group": "/ecs/h2o2",
#                 "awslogs-region": ws_settings.aws_region,
#                 "awslogs-stream-prefix": "api",
#             },
#         },
#         # 환경변수/시크릿: 필요시 아래처럼 주입
#         "environment": [
#             {"name":"RUNTIME_ENV","value":"prd"},
#             # RDS에서 동적 주입되는 값은 prd_service의 env_vars로 전달되므로 여기 생략 가능
#         ],
#         # "secrets": [ ... ]  # Agno 버전에 따라 SecretsManager 객체를 직접 참조하는 필드가 있을 수 있음
#     }],
#     skip_delete=False,
#     save_output=True,
# )

# -----------------------------------------------------------------------------
# RDS (샌드박스: t4g.micro + 퍼블릭 접근)
prd_db_subnet_group = DbSubnetGroup(
    name="h2o2-db-subnets",
    group="db",
    subnet_ids=SUBNETS_PUBLIC,  # 퍼블릭 서브넷 2개만
    skip_delete=skip_delete,
    save_output=save_output,
)

prd_db = DbInstance(
    name="h2o2-db-prd",
    group="db",
    db_name="ai",                      # 너가 쓰던 DB 이름
    port=5432,
    engine="postgres",
    engine_version="17.2",
    allocated_storage=20,              # 샌드박스 비용 절감
    db_instance_class="db.t4g.micro",  # 네가 선택한 클래스
    db_security_groups=[prd_db_sg],
    db_subnet_group=prd_db_subnet_group,
    availability_zone=ws_settings.aws_az1,  # ap-northeast-2a
    publicly_accessible=True,          # 옵션 B(내 IP만 허용) 전제
    enable_performance_insights=True,
    aws_secret=prd_db_secret,
    skip_delete=skip_delete,
    save_output=save_output,
    wait_for_delete=False,
)

# -----------------------------------------------------------------------------
# ECS Cluster
prd_ecs_cluster = EcsCluster(
    name="h2o2-cluster",               # 우리가 CLI에서 쓰던 이름과 동일
    ecs_cluster_name="h2o2-cluster",
    capacity_providers=["FARGATE"],
    skip_delete=skip_delete,
    save_output=save_output,
)

# -----------------------------------------------------------------------------
# Container ENV (앱에서 사용)
container_env = {
    "RUNTIME_ENV": "prd",
    "OPENAI_API_KEY": getenv("OPENAI_API_KEY"),
    "AGNO_MONITOR": "True",
    "AGNO_API_KEY": getenv("AGNO_API_KEY"),
    # DB 연결정보(앱 코드가 이 키들을 읽는 전제)
    "DB_HOST": AwsReference(prd_db.get_db_endpoint),
    "DB_PORT": AwsReference(prd_db.get_db_port),
    "DB_USER": AwsReference(prd_db.get_db_user),
    "DB_PASS": AwsReference(prd_db.get_db_pass),
    "DB_DATABASE": AwsReference(prd_db.get_db_name),
    # 대기/마이그 옵션
    "WAIT_FOR_DB": prd_db.enabled,
    "MIGRATE_DB": prd_db.enabled,
    # (원하면 DATABASE_URL 한 방에 쓰기)
    #"DATABASE_URL": AwsReference(prd_db.get_databse_url),
}

# -----------------------------------------------------------------------------
# 단일 서비스: h2o2-service (ALB 80 -> 8501)
# 이미지의 /app/scripts/entrypoint.sh 가 'serve' 모드로 Streamlit+FastAPI 모두 실행
prd_service = Streamlit(
    name="h2o2-service",               # ECS 서비스명
    group="app",
    image=prd_image,
    command="/app/scripts/entrypoint.sh serve",
    port_number=8501,
    ecs_task_cpu="1024",
    ecs_task_memory="2048",
    ecs_service_count=1,
    ecs_cluster=prd_ecs_cluster,
    aws_secrets=[prd_secret, prd_db_secret],
    subnets=SUBNETS_PUBLIC,            # 태스크도 퍼블릭 서브넷에서 시작(빠른 경로)
    security_groups=[prd_svc_sg],
    load_balancer_security_groups=[prd_lb_sg],
    create_load_balancer=True,
    # load_balancer_enable_https=True,
    # load_balancer_certificate_arn="arn:aws:acm:ap-northeast-2:...:certificate/...",
    health_check_path="/",             # ALB 헬스체크
    env_vars=container_env,
    skip_delete=skip_delete,
    save_output=save_output,
    wait_for_create=False,
    wait_for_delete=False,
)

# -----------------------------------------------------------------------------
# Docker/AWS Resource groups
prd_docker_resources = DockerResources(
    env=ws_settings.prd_env,
    network=ws_settings.ws_name,
    resources=[prd_image],
)

prd_aws_config = AwsResources(
    env=ws_settings.prd_env,
    apps=[prd_service],
    resources=(
        prd_lb_sg,
        prd_svc_sg,
        prd_db_sg,
        prd_secret,
        prd_db_secret,
        prd_db_subnet_group,
        prd_db,
        prd_ecs_cluster,
        prd_bucket
    ),
)
