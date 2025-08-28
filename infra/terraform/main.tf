resource "null_resource" "k3d_cluster" {
    provisioner "local-exec" {
        # had command is used to create k3d cluster , port 9092 dyal Kafka; port 9090 ta3 prometheus and port 3030 ta3 grafana | wait flag tsna 7ta ga3 lmaster node ykhdm | 8081 ta3 flink
        command= "k3d cluster create ${var.cluster_name} --servers ${var.k3d_servers} --agents ${var.k3d_agents} --port 8080:80@loadbalancer --port 8443:443@loadbalancer --port 9092:9092@loadbalancer --port 3000:3000@loadbalancer --port 9090:9090@loadbalancer --port 8081:8081@loadbalancer --wait"  
    }
     # CRITICAL: Update kubeconfig after cluster creation
    provisioner "local-exec" {
        command = "k3d kubeconfig merge ${var.cluster_name} --kubeconfig-merge-default --kubeconfig-switch-context"
    }
    provisioner "local-exec" {
        command = "mkdir -p ~/.kube"
    }
    provisioner "local-exec" {
        when = destroy
        command="k3d cluster delete iot-bigdata-cluster"
    }
}

resource "null_resource" "wait_for_k3d_cluster" {
    depends_on = [null_resource.k3d_cluster]
    # tsna 7ta ga3 l'nodes ykhdmo b7al lmaster node
    provisioner "local-exec" {
        command="kubectl wait --for=condition=Ready nodes --all --timeout=200s"
    }
}

resource "kubernetes_namespace" "namespaces" {
    for_each = toset([
        "kafka",
        "databases",
        "streaming",
        "iot-apps",
        "monitoring",
    ])
    depends_on = [ null_resource.wait_for_k3d_cluster]
    metadata {
      name = each.value
      labels = {
        managed_by = "terraform"
        environment = "production"
      }
    }
  
}