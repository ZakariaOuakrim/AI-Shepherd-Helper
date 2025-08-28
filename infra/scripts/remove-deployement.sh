#!/bin/bash 
# This script is used to remove the deployment of the IoT data platform

# remove old k3d cluster if exists
echo "Removing old k3d cluster if exists"
k3d cluster delete big-data-iot-cluster || true

# remove terraform state files and cache
echo "Removing old Terraform state files and cache"
terraform destroy 
rm -rf infra/terraform/.terraform
rm -f infra/terraform/terraform.tfstate*
rm -f infra/terraform/.terraform.lock.hcl
rm -f infra/terraform/tfplan

# Clean up any remaining docker containers from the cluster
docker ps -a --filter "name=k3d-big-data-iot-cluster" --format "table {{.Names}}" | grep -v NAMES | xargs -r docker rm -f || true

# Clean up docker networks
docker network ls --filter "name=k3d-big-data-iot-cluster" --format "{{.Name}}" | xargs -r docker network rm || true

# Clean up docker volumes
docker volume ls --filter "name=k3d-big-data-iot-cluster" --format "{{.Name}}" | xargs -r docker volume rm || true


# Clean up any temp files
rm -f *.log
rm -f /tmp/k3d-*

# Remove kubectl context
kubectl config delete-context k3d-big-data-iot-cluster || {
    echo "Warning: Could not delete kubectl context"
}
