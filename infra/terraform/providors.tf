terraform {
    required_version = ">=1.0"
    required_providers {
        docker={
            source = "kreuzwerker/docker"
            version = "3.6.2"
        }
        kubernetes = {
            source = "hashicorp/kubernetes"
            version = "~> 2.20"
        }
        helm = {
            source = "hashicorp/helm"
            version = "~> 2.9"
        }
    }
}

provider "docker" {}

provider "kubernetes" { 
  config_path = "~/.kube/config"
}

provider "helm" {
  kubernetes {
    config_path = "~/.kube/config"
  }
}