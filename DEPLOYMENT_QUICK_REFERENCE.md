# Deployment Quick Reference

Quick reference for common deployment commands and operations.

## Docker Commands

### Build & Start

```bash
# Build production image
./scripts/build.sh

# Start production containers
docker-compose up -d

# Start development environment
./scripts/dev.sh start

# Build specific service
docker-compose build hrl-finance
```

### Stop & Remove

```bash
# Stop containers
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Stop development environment
./scripts/dev.sh stop
```

### Logs & Monitoring

```bash
# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f hrl-finance

# View last 100 lines
docker-compose logs --tail=100

# Development logs
./scripts/dev.sh logs
```

### Container Management

```bash
# List running containers
docker-compose ps

# Restart containers
docker-compose restart

# Execute command in container
docker-compose exec hrl-finance bash

# View container stats
docker stats
```

## Deployment Scripts

### Build Script

```bash
# Build with default settings
./scripts/build.sh

# Build with version tag
VERSION=1.0.0 ./scripts/build.sh
```

### Deploy Script

```bash
# Deploy to production
./scripts/deploy.sh production deploy

# Check status
./scripts/deploy.sh production status

# View logs
./scripts/deploy.sh production logs

# Rollback
./scripts/deploy.sh production rollback
```

### Development Script

```bash
# Start development
./scripts/dev.sh start

# Stop development
./scripts/dev.sh stop

# Restart development
./scripts/dev.sh restart

# View logs
./scripts/dev.sh logs

# Open shell
./scripts/dev.sh shell

# Run tests
./scripts/dev.sh test

# Clean up
./scripts/dev.sh clean
```

## Health Checks

```bash
# Check health endpoint
curl http://localhost:8000/health

# Check with details
curl -v http://localhost:8000/health

# Check from inside container
docker-compose exec hrl-finance curl http://localhost:8000/health
```

## Nginx Commands

### Start with Nginx

```bash
# Start with Nginx reverse proxy
docker-compose -f nginx/docker-compose.nginx.yml up -d

# View Nginx logs
docker-compose -f nginx/docker-compose.nginx.yml logs -f nginx

# Reload Nginx config
docker-compose -f nginx/docker-compose.nginx.yml exec nginx nginx -s reload

# Test Nginx config
docker-compose -f nginx/docker-compose.nginx.yml exec nginx nginx -t
```

### SSL Certificates

```bash
# Obtain Let's Encrypt certificate
docker-compose -f nginx/docker-compose.nginx.yml run --rm certbot certonly \
  --webroot --webroot-path=/var/www/certbot \
  --email your-email@example.com \
  --agree-tos -d yourdomain.com

# Renew certificates
docker-compose -f nginx/docker-compose.nginx.yml run --rm certbot renew

# Test renewal
docker-compose -f nginx/docker-compose.nginx.yml run --rm certbot renew --dry-run
```

## Kubernetes Commands

### Deploy

```bash
# Create namespace
kubectl create namespace hrl-finance

# Apply manifests
kubectl apply -f k8s/deployment.yaml -n hrl-finance
kubectl apply -f k8s/ingress.yaml -n hrl-finance

# Check deployment
kubectl get all -n hrl-finance
```

### Monitor

```bash
# View pods
kubectl get pods -n hrl-finance

# View logs
kubectl logs -f deployment/hrl-finance -n hrl-finance

# Describe pod
kubectl describe pod <pod-name> -n hrl-finance

# Check events
kubectl get events -n hrl-finance --sort-by='.lastTimestamp'
```

### Scale

```bash
# Scale manually
kubectl scale deployment hrl-finance --replicas=3 -n hrl-finance

# Autoscale
kubectl autoscale deployment hrl-finance \
  --cpu-percent=70 --min=2 --max=10 -n hrl-finance
```

### Update

```bash
# Update image
kubectl set image deployment/hrl-finance \
  hrl-finance=your-registry/hrl-finance:v2 -n hrl-finance

# Rollout status
kubectl rollout status deployment/hrl-finance -n hrl-finance

# Rollback
kubectl rollout undo deployment/hrl-finance -n hrl-finance
```

## Backup & Restore

### Backup

```bash
# Backup models
docker run --rm -v hrl-finance_models:/data -v $(pwd):/backup \
  alpine tar czf /backup/models-backup.tar.gz -C /data .

# Backup configs
docker run --rm -v hrl-finance_configs:/data -v $(pwd):/backup \
  alpine tar czf /backup/configs-backup.tar.gz -C /data .

# Backup all volumes
for vol in models configs reports results; do
  docker run --rm -v hrl-finance_${vol}:/data -v $(pwd):/backup \
    alpine tar czf /backup/${vol}-backup.tar.gz -C /data .
done
```

### Restore

```bash
# Restore models
docker run --rm -v hrl-finance_models:/data -v $(pwd):/backup \
  alpine tar xzf /backup/models-backup.tar.gz -C /data

# Restore configs
docker run --rm -v hrl-finance_configs:/data -v $(pwd):/backup \
  alpine tar xzf /backup/configs-backup.tar.gz -C /data
```

## Troubleshooting

### Check Container Status

```bash
# View all containers
docker ps -a

# View container details
docker inspect hrl-finance-app

# View container logs
docker logs hrl-finance-app

# Check resource usage
docker stats hrl-finance-app
```

### Network Issues

```bash
# List networks
docker network ls

# Inspect network
docker network inspect hrl-network

# Test connectivity
docker-compose exec hrl-finance ping google.com

# Test backend from frontend
docker-compose exec hrl-finance curl http://localhost:8000/health
```

### Volume Issues

```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect hrl-finance_models

# Check volume contents
docker run --rm -v hrl-finance_models:/data alpine ls -la /data
```

### Clean Up

```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune

# Remove unused volumes
docker volume prune

# Remove everything
docker system prune -a --volumes
```

## Environment Variables

### View Current Settings

```bash
# View all environment variables
docker-compose exec hrl-finance env

# View specific variable
docker-compose exec hrl-finance printenv PORT

# View .env file
cat .env
```

### Update Settings

```bash
# Edit .env file
nano .env

# Restart to apply changes
docker-compose restart
```

## Performance Monitoring

### Resource Usage

```bash
# View container stats
docker stats

# View specific container
docker stats hrl-finance-app

# View top processes
docker-compose exec hrl-finance top
```

### API Performance

```bash
# Test response time
time curl http://localhost:8000/health

# Detailed timing
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health

# Load testing (with Apache Bench)
ab -n 1000 -c 10 http://localhost:8000/health
```

## Database Operations (if applicable)

### Backup Database

```bash
# PostgreSQL backup
docker-compose exec postgres pg_dump -U user dbname > backup.sql

# SQLite backup
docker-compose exec hrl-finance cp /app/data.db /backup/data.db
```

### Restore Database

```bash
# PostgreSQL restore
docker-compose exec -T postgres psql -U user dbname < backup.sql

# SQLite restore
docker-compose exec hrl-finance cp /backup/data.db /app/data.db
```

## Security

### Update Dependencies

```bash
# Update Python packages
docker-compose exec hrl-finance pip install --upgrade -r requirements.txt

# Update Node packages
docker-compose exec hrl-finance sh -c "cd frontend && npm update"

# Rebuild image
./scripts/build.sh
```

### Scan for Vulnerabilities

```bash
# Scan Docker image
docker scan hrl-finance:latest

# Scan with Trivy
trivy image hrl-finance:latest
```

## Useful One-Liners

```bash
# Quick health check
curl -f http://localhost:8000/health && echo "OK" || echo "FAIL"

# Watch logs in real-time
watch -n 1 'docker-compose logs --tail=20'

# Count running containers
docker ps | grep hrl-finance | wc -l

# Get container IP
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' hrl-finance-app

# Export logs to file
docker-compose logs > logs-$(date +%Y%m%d-%H%M%S).txt

# Find large files in volumes
docker run --rm -v hrl-finance_models:/data alpine du -sh /data/*

# Check disk usage
docker system df

# View image history
docker history hrl-finance:latest
```

## Emergency Procedures

### Service Down

```bash
# 1. Check status
docker-compose ps

# 2. View logs
docker-compose logs --tail=100

# 3. Restart service
docker-compose restart

# 4. If still down, rebuild
docker-compose down
./scripts/build.sh
docker-compose up -d
```

### Out of Disk Space

```bash
# 1. Check disk usage
df -h
docker system df

# 2. Clean up
docker system prune -a --volumes

# 3. Remove old images
docker images | grep '<none>' | awk '{print $3}' | xargs docker rmi

# 4. Clear logs
truncate -s 0 /var/lib/docker/containers/*/*-json.log
```

### Memory Issues

```bash
# 1. Check memory usage
docker stats --no-stream

# 2. Restart container
docker-compose restart

# 3. Increase memory limit in docker-compose.yml
# mem_limit: 4g

# 4. Restart with new limits
docker-compose up -d
```

## Support

For detailed information, see:
- [Deployment Guide](DEPLOYMENT.md)
- [Deployment Checklist](DEPLOYMENT_CHECKLIST.md)
- [API Documentation](backend/API_DOCUMENTATION.md)
