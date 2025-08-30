#!/bin/bash

# AI Shepherd - Sheep Tracking System Deployment Script
# This script deploys the complete sheep tracking system to Kubernetes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if kubectl is available
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if we can connect to Kubernetes cluster
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Create namespaces
create_namespaces() {
    print_status "Creating namespaces..."
    
    local namespaces=("kafka" "databases" "iot-apps" "monitoring" "streaming")
    
    for ns in "${namespaces[@]}"; do
        kubectl create namespace "$ns" --dry-run=client -o yaml | kubectl apply -f -
        print_status "Namespace '$ns' created/updated"
    done
    
    print_success "All namespaces created"
}

# Deploy the web dashboard
deploy_web_dashboard() {
    print_status "Deploying AI Shepherd Web Dashboard..."
    
    # Apply the web dashboard YAML
    kubectl apply -f - <<EOF
$(cat << 'YAML_END'
# Insert the fixed web-dashboard.yaml content here
# This would be the content from the first artifact
apiVersion: apps/v1
kind: Deployment
metadata:
  name: websocket-server
  namespace: iot-apps
spec:
  replicas: 1
  selector:
    matchLabels:
      app: websocket-server
  template:
    metadata:
      labels:
        app: websocket-server
    spec:
      containers:
      - name: websocket-server
        image: python:3.11-slim
        ports:
        - containerPort: 8080
        command: ["/bin/bash"]
        args:
        - -c
        - |
          pip install kafka-python websockets numpy
          cd /app
          python websocket_server.py
        volumeMounts:
        - name: app-code
          mountPath: /app
        env:
        - name: PYTHONUNBUFFERED
          value: "1"
        resources:
          limits:
            memory: "512Mi"
            cpu: "500m"
          requests:
            memory: "256Mi"
            cpu: "250m"
      volumes:
      - name: app-code
        configMap:
          name: websocket-server-code
YAML_END
)
EOF

    print_success "Web Dashboard deployment started"
}

# Wait for deployments to be ready
wait_for_deployments() {
    print_status "Waiting for deployments to be ready..."
    
    local deployments=("websocket-server" "web-dashboard")
    
    for deployment in "${deployments[@]}"; do
        print_status "Waiting for $deployment to be ready..."
        kubectl wait --for=condition=available --timeout=300s deployment/$deployment -n iot-apps
        print_success "$deployment is ready"
    done
}

# Start port forwarding
start_port_forward() {
    print_status "Starting port forwarding for web dashboard..."
    
    # Kill any existing port forwarding
    pkill -f "kubectl port-forward.*web-dashboard-service" || true
    
    # Start port forwarding in background
    kubectl port-forward svc/web-dashboard-service -n iot-apps 8080:80 &
    local PORT_FORWARD_PID=$!
    
    sleep 3  # Wait for port forward to establish
    
    if kill -0 $PORT_FORWARD_PID 2>/dev/null; then
        print_success "Port forwarding started (PID: $PORT_FORWARD_PID)"
        echo $PORT_FORWARD_PID > /tmp/sheep-tracker-port-forward.pid
    else
        print_error "Failed to start port forwarding"
        return 1
    fi
}

# Create test producer
create_test_producer() {
    print_status "Creating test data producer..."
    
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: test-producer-script
  namespace: iot-apps
data:
  test_producer.py: |
    #!/usr/bin/env python3
    import json
    import time
    import random
    from kafka import KafkaProducer
    from datetime import datetime
    
    try:
        producer = KafkaProducer(
            bootstrap_servers=[
                'kafka-0.kafka-headless.kafka.svc.cluster.local:9092',
                'kafka-1.kafka-headless.kafka.svc.cluster.local:9092'
            ],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        # Base positions around Morocco coordinates
        sheep_positions = {
            'Sheep_1': {'lat': 31.010530, 'lon': -7.012203},  # Isolated sheep
            'Sheep_2': {'lat': 31.000002, 'lon': -7.000008},
            'Sheep_3': {'lat': 31.000015, 'lon': -7.000012},
            'Sheep_4': {'lat': 30.999998, 'lon': -7.000005},
            'Sheep_5': {'lat': 31.000020, 'lon': -6.999995}
        }
        
        print("Starting sheep data producer...")
        
        while True:
            for sheep_id, pos in sheep_positions.items():
                # Add random movement
                new_lat = pos['lat'] + (random.random() - 0.5) * 0.0001
                new_lon = pos['lon'] + (random.random() - 0.5) * 0.0001
                
                # Update base position slightly for next iteration
                pos['lat'] = new_lat
                pos['lon'] = new_lon
                
                message = {
                    'sheep_id': sheep_id,
                    'latitude': new_lat,
                    'longitude': new_lon,
                    'timestamp': datetime.now().isoformat()
                }
                
                producer.send('sheep-topic', message)
                print(f"Sent: {sheep_id} at ({new_lat:.6f}, {new_lon:.6f})")
                
                time.sleep(0.5)  # Small delay between sheep
            
            time.sleep(2)  # Wait before next round
            
    except Exception as e:
        print(f"Error: {e}")
        print("Kafka not available - this is normal if Kafka isn't deployed")
        time.sleep(30)
EOF

    print_success "Test producer script created"
}

# Display status and access information
show_status() {
    print_success "AI Shepherd - Sheep Tracking System Deployed Successfully!"
    echo ""
    echo "📊 System Status:"
    echo "=================="
    
    # Check deployment status
    echo ""
    echo "Deployments:"
    kubectl get deployments -n iot-apps
    
    echo ""
    echo "Services:"
    kubectl get services -n iot-apps
    
    echo ""
    echo "🌐 Access Information:"
    echo "======================"
    echo "Web Dashboard: http://localhost:8080"
    echo ""
    echo "🔧 Management Commands:"
    echo "======================="
    echo "View WebSocket logs:    kubectl logs -f deployment/websocket-server -n iot-apps"
    echo "View Dashboard logs:    kubectl logs -f deployment/web-dashboard -n iot-apps"
    echo "Stop port forwarding:   kill \$(cat /tmp/sheep-tracker-port-forward.pid)"
    echo "Restart port forward:   kubectl port-forward svc/web-dashboard-service -n iot-apps 8080:80"
    echo ""
    echo "🧪 Testing:"
    echo "==========="
    echo "Start test producer:"
    echo "  kubectl run test-producer -n iot-apps --image=python:3.11-slim --rm -it -- /bin/bash"
    echo "  # Then inside the pod:"
    echo "  pip install kafka-python"
    echo "  kubectl get configmap test-producer-script -n iot-apps -o jsonpath='{.data.test_producer\\.py}' > test_producer.py"
    echo "  python test_producer.py"
}

# Cleanup function
cleanup() {
    print_status "Cleaning up resources..."
    
    # Stop port forwarding
    if [ -f /tmp/sheep-tracker-port-forward.pid ]; then
        local PID=$(cat /tmp/sheep-tracker-port-forward.pid)
        kill $PID 2>/dev/null || true
        rm -f /tmp/sheep-tracker-port-forward.pid
    fi
    
    # Optionally delete the deployment
    if [ "$1" == "--delete" ]; then
        print_warning "Deleting all sheep tracker resources..."
        kubectl delete namespace iot-apps --ignore-not-found=true
        print_success "Resources deleted"
    fi
}

# Handle script interruption
trap cleanup EXIT

# Main execution
main() {
    echo "🐑 AI Shepherd - Sheep Tracking System Deployment"
    echo "=================================================="
    echo ""
    
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            create_namespaces
            deploy_web_dashboard
            wait_for_deployments
            create_test_producer
            start_port_forward
            show_status
            
            echo ""
            print_success "Deployment complete! Your sheep tracking system is ready."
            print_status "Press Ctrl+C to stop port forwarding and exit"
            
            # Keep the script running to maintain port forwarding
            while true; do
                sleep 30
                if ! kill -0 $(cat /tmp/sheep-tracker-port-forward.pid 2>/dev/null) 2>/dev/null; then
                    print_warning "Port forwarding stopped, restarting..."
                    start_port_forward
                fi
            done
            ;;
            
        "status")
            show_status
            ;;
            
        "cleanup")
            cleanup --delete
            ;;
            
        "help"|*)
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  deploy   - Deploy the sheep tracking system (default)"
            echo "  status   - Show system status"
            echo "  cleanup  - Remove all deployed resources"
            echo "  help     - Show this help message"
            ;;
    esac
}

# Execute main function with all arguments
main "$@"