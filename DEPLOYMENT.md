# HRL Finance System - Deployment Guide

This guide covers deployment options for the HRL Finance System, including Docker containerization, local development, and production deployment.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Development Deployment](#development-deployment)
- [Production Deployment](#production-deployment)
- [Environment Configuration](#environment-configuration)
- [Docker Commands](#docker-commands)
- [Troubleshooting](#troubleshooting)
- [Monitoring and Maintenance](#monitoring-and-maintenance)

## Prerequisites

### Required Software

- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **Git**: For cloning the repository

### System Requirements

**Minimum:**
- CPU: 2 cores
- RAM: 4 GB
- Disk: 10 GB free space

**Recommended:**
- CPU: 4+ cores
- RAM: 8+ GB
- Disk: 20+ GB free space

### Installation

#### Docker Installation

**macOS:**
```bash
# Install Docker Desktop
brew install --cask docker
```

**Linux (Ubuntu/Debian):**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt-get update
sudo apt-get install docker-compose-plugin
```

**Windows:**
Download and install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/hrl-finance-system.git
cd hrl-finance-system
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings (optional for development)
nano .env
```

### 3. Build and Run

```bash
# Build the Docker image
./scripts/build.sh

# Start the application
docker-compose up -d

# Check status
docker-compose ps
```

### 4. Access the Application

- **Frontend**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Development Deployment

Development mode provides hot-reload for both backend and frontend, making it ideal for active development.

### Start Development Environment

```bash
# Start development containers
./scripts/dev.sh start

# Or manually with docker-compose
docker-compose --profile dev up -d
```

This will start:
- Backend API on http://localhost:8000 (with hot-reload)
- Frontend dev server on http://localhost:5173 (with hot-reload)

### Development Commands

```bash
# View logs
./scripts/dev.sh logs

# Restart services
./scripts/dev.sh restart

# Open shell in container
./scripts/dev.sh shell

# Run tests
./scripts/dev.sh test

# Stop development environment
./scripts/dev.sh stop

# Clean up (removes volumes)
./scripts/dev.sh clean
```

### Local Development (Without Docker)

If you prefer to run without Docker:

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:socket_app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Production Deployment

### 1. Prepare Environment

```bash
# Copy and configure environment file
cp .env.example .env

# Edit production settings
nano .env
```

**Important Production Settings:**

```bash
# Set specific CORS origins
CORS_ORIGINS=https://yourdomain.com

# Enable production optimizations
LOG_LEVEL=WARNING
WORKERS=4

# Configure paths
CONFIGS_DIR=/app/configs
MODELS_DIR=/app/models
REPORTS_DIR=/app/reports
```

### 2. Build Production Image

```bash
# Build with version tag
VERSION=1.0.0 ./scripts/build.sh

# Or manually
docker build -t hrl-finance:1.0.0 -f Dockerfile .
docker tag hrl-finance:1.0.0 hrl-finance:latest
```

### 3. Deploy

```bash
# Deploy using script
./scripts/deploy.sh production deploy

# Or manually
docker-compose up -d
```

### 4. Verify Deployment

```bash
# Check service status
./scripts/deploy.sh production status

# Check health endpoint
curl http://localhost:8000/health

# View logs
docker-compose logs -f
```

## Environment Configuration

### Core Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `PORT` | Server port | 8000 | No |
| `HOST` | Server host | 0.0.0.0 | No |
| `CORS_ORIGINS` | Allowed CORS origins | * | Yes (prod) |

### File Paths

| Variable | Description | Default |
|----------|-------------|---------|
| `CONFIGS_DIR` | Scenario configurations | ./configs |
| `MODELS_DIR` | Trained models | ./models |
| `REPORTS_DIR` | Generated reports | ./reports |
| `RESULTS_DIR` | Simulation results | ./results |
| `RUNS_DIR` | TensorBoard logs | ./runs |
| `LOGS_DIR` | Application logs | ./logs |

### Training Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `MAX_TRAINING_EPISODES` | Max training episodes | 10000 |
| `DEFAULT_SAVE_INTERVAL` | Checkpoint interval | 100 |
| `DEFAULT_EVAL_EPISODES` | Evaluation episodes | 10 |

### WebSocket Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `WEBSOCKET_PING_TIMEOUT` | Ping timeout (seconds) | 60 |
| `WEBSOCKET_PING_INTERVAL` | Ping interval (seconds) | 25 |

### Performance Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `WORKERS` | Number of worker processes | 4 |
| `MAX_REQUESTS` | Max requests per worker | 1000 |

## Docker Commands

### Container Management

```bash
# Start containers
docker-compose up -d

# Stop containers
docker-compose down

# Restart containers
docker-compose restart

# View running containers
docker-compose ps

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f hrl-finance
```

### Image Management

```bash
# List images
docker images | grep hrl-finance

# Remove old images
docker image prune -f

# Tag image
docker tag hrl-finance:latest hrl-finance:1.0.0

# Push to registry (if configured)
docker push your-registry/hrl-finance:1.0.0
```

### Volume Management

```bash
# List volumes
docker volume ls

# Backup volumes
docker run --rm -v hrl-finance_models:/data -v $(pwd):/backup \
  alpine tar czf /backup/models-backup.tar.gz -C /data .

# Restore volumes
docker run --rm -v hrl-finance_models:/data -v $(pwd):/backup \
  alpine tar xzf /backup/models-backup.tar.gz -C /data
```

### Debugging

```bash
# Execute command in running container
docker-compose exec hrl-finance bash

# View container resource usage
docker stats

# Inspect container
docker inspect hrl-finance-app

# View container logs with timestamps
docker-compose logs -f --timestamps
```

## Troubleshooting

### Common Issues

#### 1. Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or change port in .env
PORT=8001
```

#### 2. Container Won't Start

```bash
# Check logs
docker-compose logs hrl-finance

# Check container status
docker-compose ps

# Rebuild image
docker-compose build --no-cache
```

#### 3. Permission Errors

```bash
# Fix volume permissions
sudo chown -R $USER:$USER ./models ./reports ./results

# Or run with proper user in docker-compose.yml
user: "${UID}:${GID}"
```

#### 4. Out of Memory

```bash
# Increase Docker memory limit in Docker Desktop settings
# Or add to docker-compose.yml:
services:
  hrl-finance:
    mem_limit: 4g
```

#### 5. Frontend Not Loading

```bash
# Check if frontend was built
docker-compose exec hrl-finance ls -la /app/frontend/dist

# Rebuild if necessary
./scripts/build.sh
```

### Health Check Failures

```bash
# Check health endpoint manually
curl http://localhost:8000/health

# Check if Python dependencies are installed
docker-compose exec hrl-finance pip list

# Check if backend is running
docker-compose exec hrl-finance ps aux | grep uvicorn
```

### Network Issues

```bash
# Check Docker network
docker network ls
docker network inspect hrl-network

# Recreate network
docker-compose down
docker network prune
docker-compose up -d
```

## Monitoring and Maintenance

### Health Monitoring

```bash
# Automated health check
while true; do
  curl -f http://localhost:8000/health || echo "Health check failed"
  sleep 60
done
```

### Log Management

```bash
# View recent logs
docker-compose logs --tail=100

# Follow logs
docker-compose logs -f

# Export logs
docker-compose logs > logs-$(date +%Y%m%d).txt

# Rotate logs (add to cron)
docker-compose logs --tail=1000 > logs-archive.txt
docker-compose restart
```

### Backup Strategy

```bash
# Backup script
#!/bin/bash
BACKUP_DIR="./backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup volumes
docker run --rm -v hrl-finance_models:/data -v $(pwd)/$BACKUP_DIR:/backup \
  alpine tar czf /backup/models.tar.gz -C /data .

docker run --rm -v hrl-finance_configs:/data -v $(pwd)/$BACKUP_DIR:/backup \
  alpine tar czf /backup/configs.tar.gz -C /data .

docker run --rm -v hrl-finance_results:/data -v $(pwd)/$BACKUP_DIR:/backup \
  alpine tar czf /backup/results.tar.gz -C /data .

echo "Backup completed: $BACKUP_DIR"
```

### Updates and Upgrades

```bash
# Pull latest code
git pull

# Rebuild and deploy
./scripts/deploy.sh production deploy

# Or with zero-downtime (if using load balancer)
docker-compose up -d --no-deps --build hrl-finance
```

### Performance Monitoring

```bash
# Monitor resource usage
docker stats hrl-finance-app

# Monitor API performance
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health

# curl-format.txt:
time_namelookup:  %{time_namelookup}\n
time_connect:  %{time_connect}\n
time_starttransfer:  %{time_starttransfer}\n
time_total:  %{time_total}\n
```

## Production Best Practices

### Security

1. **Use HTTPS**: Configure reverse proxy (nginx/traefik) with SSL
2. **Restrict CORS**: Set specific origins in production
3. **Environment Variables**: Never commit .env files
4. **Regular Updates**: Keep Docker images and dependencies updated
5. **Network Isolation**: Use Docker networks for service isolation

### Scalability

1. **Horizontal Scaling**: Run multiple containers behind load balancer
2. **Resource Limits**: Set memory and CPU limits
3. **Caching**: Implement Redis for session/data caching
4. **Database**: Use PostgreSQL for metadata instead of file system

### Reliability

1. **Health Checks**: Configure proper health check intervals
2. **Restart Policy**: Use `restart: unless-stopped`
3. **Logging**: Centralize logs with ELK stack or similar
4. **Monitoring**: Use Prometheus + Grafana for metrics
5. **Backups**: Automated daily backups of volumes

## Support

For issues or questions:
- Check the [Troubleshooting](#troubleshooting) section
- Review logs: `docker-compose logs -f`
- Open an issue on GitHub
- Consult the API documentation: http://localhost:8000/docs

## License

MIT License - See LICENSE file for details
