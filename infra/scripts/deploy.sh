#!/bin/bash

set -e

echo "Starting deployment script for herd sheep data platform"

# Terraform deployment
cd infra/terraform || exit 1
terraform init
terraform plan -out=tfplan
terraform apply tfplan
cd ../..

pip install kubernetes

echo "Building kafka producer"
cd docker/sheep
docker build --no-cache -t sheep-producer:latest .
k3d image import sheep-producer:latest --cluster big-data-iot-cluster
cd ../..

echo "Building kafka consumer"
cd docker/knn
docker build --no-cache -t knn:latest .
k3d image import knn:latest --cluster big-data-iot-cluster
cd ../..


ansible-playbook -i infra/ansible/inventory/hosts.yaml infra/ansible/playbooks/setup-cluster.yaml


echo "checking deployments"
kubectl get pods --all-namespaces
