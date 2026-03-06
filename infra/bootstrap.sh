#!/usr/bin/env bash
# bootstrap.sh — provisiona a infraestrutura base via CloudFormation.
# Execute UMA VEZ antes de usar o GitHub Actions.
#
# Pre-requisito: AWS CLI configurado com permissoes de admin
#
# Uso:
#   export AWS_REGION=us-east-1
#   export GITHUB_ORG=seu-usuario
#   export GITHUB_REPO=nome-do-repositorio
#   bash infra/bootstrap.sh

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
GITHUB_ORG="${GITHUB_ORG:?Defina GITHUB_ORG}"
GITHUB_REPO="${GITHUB_REPO:?Defina GITHUB_REPO}"
STACK_NAME="passos-magicos-infra"

echo "==> Deployando stack CloudFormation: $STACK_NAME"
echo "==> Regiao: $REGION | Repo: $GITHUB_ORG/$GITHUB_REPO"
echo ""

aws cloudformation deploy \
  --template-file infra/cloudformation.yml \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    GitHubOrg="$GITHUB_ORG" \
    GitHubRepo="$GITHUB_REPO" \
  --no-fail-on-empty-changeset

echo ""
echo "==> Stack deployada. Outputs:"
echo ""

aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --query "Stacks[0].Outputs[*].[OutputKey, OutputValue]" \
  --output table

echo ""
echo "================================================================"
echo "Adicione os seguintes secrets no GitHub:"
echo "  Repositorio > Settings > Secrets and variables > Actions"
echo ""

GITHUB_ROLE=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" --region "$REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='GitHubActionsRoleArn'].OutputValue" \
  --output text)

SAGEMAKER_ROLE=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" --region "$REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='SageMakerExecutionRoleArn'].OutputValue" \
  --output text)

echo "  AWS_ROLE_ARN       = $GITHUB_ROLE"
echo "  SAGEMAKER_ROLE_ARN = $SAGEMAKER_ROLE"
echo "================================================================"
