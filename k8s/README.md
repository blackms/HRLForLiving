# Kubernetes Deployment

This directory contains Kubernetes manifests for deploying the HRL Finance System to a Kubernetes cluster.

## Prerequisites

- Kubernetes cluster (v1.20+)
- kubectl configured to access your cluster
- Persistent volume provisioner (for storage)
- Ingress controller (nginx recommended)
- cert-manager (optional, for SSL certificates)

## Quick Start

### 1. Build and Push Docker Image

```bash
# Build image
docker build -t your-registry/hrl-finance:latest -f Dockerfile .

# Push to registry
docker push your-registry/hrl-finance:latest
```

### 2. Update Image Reference

Edit `deployment.yaml` and update the image reference:

```yaml
spec:
  containers:
  - name: hrl-finance
    image: your-registry/hrl-finance:latest
```

### 3. Configure Settings

Edit `deployment.yaml` ConfigMap section to set your environment variables:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: hrl-finance-config
data:
  cors_origins: "https://yourdomain.com"
  log_level: "INFO"
```

### 4. Deploy to Kubernetes

```bash
# Create namespace (optional)
kubectl create namespace hrl-finance

# Apply manifests
kubectl apply -f k8s/deployment.yaml -n hrl-finance

# Apply ingress (if using)
kubectl apply -f k8s/ingress.yaml -n hrl-finance
```

### 5. Verify Deployment

```bash
# Check pods
kubectl get pods -n hrl-finance

# Check service
kubectl get svc -n hrl-finance

# Check ingress
kubectl get ingress -n hrl-finance

# View logs
kubectl logs -f deployment/hrl-finance -n hrl-finance
```

## Configuration

### Storage Classes

The manifests use `standard` storage class by default. Update if your cluster uses a different storage class:

```yaml
spec:
  storageClassName: your-storage-class
```

### Resource Limits

Adjust resource requests and limits based on your workload:

```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

### Replicas

For high availability, increase the number of replicas:

```yaml
spec:
  replicas: 3
```

### Ingress

Update the ingress hostname in `ingress.yaml`:

```yaml
spec:
  tls:
  - hosts:
    - hrl-finance.yourdomain.com
  rules:
  - host: hrl-finance.yourdomain.com
```

## Scaling

### Manual Scaling

```bash
# Scale to 3 replicas
kubectl scale deployment hrl-finance --replicas=3 -n hrl-finance
```

### Horizontal Pod Autoscaler

Create an HPA for automatic scaling:

```bash
kubectl autoscale deployment hrl-finance \
  --cpu-percent=70 \
  --min=2 \
  --max=10 \
  -n hrl-finance
```

## Monitoring

### View Logs

```bash
# All pods
kubectl logs -f deployment/hrl-finance -n hrl-finance

# Specific pod
kubectl logs -f pod/hrl-finance-xxx -n hrl-finance

# Previous container logs
kubectl logs --previous pod/hrl-finance-xxx -n hrl-finance
```

### Check Status

```bash
# Deployment status
kubectl rollout status deployment/hrl-finance -n hrl-finance

# Pod status
kubectl get pods -n hrl-finance -o wide

# Events
kubectl get events -n hrl-finance --sort-by='.lastTimestamp'
```

### Execute Commands

```bash
# Open shell in pod
kubectl exec -it deployment/hrl-finance -n hrl-finance -- /bin/bash

# Run command
kubectl exec deployment/hrl-finance -n hrl-finance -- python --version
```

## Updates

### Rolling Update

```bash
# Update image
kubectl set image deployment/hrl-finance \
  hrl-finance=your-registry/hrl-finance:v2 \
  -n hrl-finance

# Or apply updated manifest
kubectl apply -f k8s/deployment.yaml -n hrl-finance
```

### Rollback

```bash
# View rollout history
kubectl rollout history deployment/hrl-finance -n hrl-finance

# Rollback to previous version
kubectl rollout undo deployment/hrl-finance -n hrl-finance

# Rollback to specific revision
kubectl rollout undo deployment/hrl-finance --to-revision=2 -n hrl-finance
```

## Backup and Restore

### Backup Persistent Volumes

```bash
# Create backup pod
kubectl run backup --image=alpine --rm -it \
  --overrides='
{
  "spec": {
    "containers": [{
      "name": "backup",
      "image": "alpine",
      "command": ["/bin/sh"],
      "stdin": true,
      "tty": true,
      "volumeMounts": [{
        "name": "models",
        "mountPath": "/data/models"
      }]
    }],
    "volumes": [{
      "name": "models",
      "persistentVolumeClaim": {
        "claimName": "hrl-finance-models-pvc"
      }
    }]
  }
}' -n hrl-finance

# Inside the pod, create backup
tar czf /tmp/models-backup.tar.gz -C /data/models .
```

## Troubleshooting

### Pod Not Starting

```bash
# Describe pod
kubectl describe pod hrl-finance-xxx -n hrl-finance

# Check events
kubectl get events -n hrl-finance

# Check logs
kubectl logs pod/hrl-finance-xxx -n hrl-finance
```

### Storage Issues

```bash
# Check PVCs
kubectl get pvc -n hrl-finance

# Describe PVC
kubectl describe pvc hrl-finance-models-pvc -n hrl-finance

# Check PVs
kubectl get pv
```

### Network Issues

```bash
# Test service connectivity
kubectl run test --image=curlimages/curl --rm -it -- \
  curl http://hrl-finance-service/health

# Check service endpoints
kubectl get endpoints hrl-finance-service -n hrl-finance
```

## Cleanup

```bash
# Delete all resources
kubectl delete -f k8s/ -n hrl-finance

# Delete namespace
kubectl delete namespace hrl-finance

# Delete PVCs (if needed)
kubectl delete pvc --all -n hrl-finance
```

## Production Considerations

1. **Use Secrets**: Store sensitive data in Kubernetes Secrets
2. **Resource Quotas**: Set namespace resource quotas
3. **Network Policies**: Implement network policies for security
4. **Pod Security**: Use Pod Security Standards
5. **Monitoring**: Deploy Prometheus and Grafana
6. **Logging**: Use ELK stack or similar for centralized logging
7. **Backup**: Implement automated backup solution (Velero)
8. **High Availability**: Use multiple replicas and pod anti-affinity

## Support

For Kubernetes-specific issues, consult:
- Kubernetes documentation: https://kubernetes.io/docs/
- kubectl cheat sheet: https://kubernetes.io/docs/reference/kubectl/cheatsheet/
