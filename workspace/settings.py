from pathlib import Path

from agno.workspace.settings import WorkspaceSettings

#
# We define workspace settings using a WorkspaceSettings object
# these values can also be set using environment variables
# Import them into your project using `from workspace.settings import ws_settings`
#
ws_settings = WorkspaceSettings(
    # Workspace name
    ws_name="agent-app",
    # Path to the workspace root
    ws_root=Path(__file__).parent.parent.resolve(),
    # -*- Workspace Environments
    dev_env="dev",
    prd_env="prd",
    # default env for `agno ws` commands
    default_env="dev",
    # -*- Image Settings
    #--------------------------
    # Repository for images
    # - local for dev, aws for prd
    # dev <-> prd: change image_repo & push_images
    #--------------------------
    image_repo="037129617559.dkr.ecr.ap-northeast-2.amazonaws.com/h2o2",
    #image_repo="local",
    # 'Name:tag' for the image
    image_name="agent-app",
    # Build images locally
    #build_images=False,
    build_images=True,
    # Push images to the registry. True for prd, False for dev
    #push_images=False,
    push_images=True,
    # Skip cache when building images
    skip_image_cache=False,
    # Force pull images
    force_pull_images=False,
    # -*- AWS settings
    # Region for AWS resources
    aws_region="ap-northeast-2",
    # Availability Zones for AWS resources
    aws_az1="ap-northeast-2a",
    aws_az2="ap-northeast-2b",
    # Subnets for AWS resources
    subnet_ids=["subnet-08e75869bc55b5df1", "subnet-095f958fbf81ed9d4", "subnet-0f55d4128b7e52e0a", "subnet-08d15ceb04b65afdd"],
    # Security Groups for AWS resources
    # aws_security_group_ids=["sg-xyz", "sg-xyz"],
)
