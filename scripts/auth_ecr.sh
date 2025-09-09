#!/bin/bash

AWS_REGION=ap-northeast-2
AWS_ACCOUNT_ID=037129617559

set -e

# Authenticate with ecr
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
