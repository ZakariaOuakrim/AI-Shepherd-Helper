variable "cluster_name" {
  description = "Name of k3d cluster"
  type = string
  default = "big-data-iot-cluster"
}

variable "k3d_servers" {
  description = "Number of master servers in the k3d cluster"
  type = number
  default = 1
}

variable "k3d_agents" {
  description = "Number of slave nodes in the k3d cluster"
  type = number
  default = 1
}

variable "kafka_replicas" {
  description = "Number of Kafka replicas"
  type = number
  default = 2
}

variable "enable_monitoring" {
  description = "Enable monitoring with Prometheus and Grafana"
  type = bool
  default = true
  
}
