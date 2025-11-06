#!/bin/bash
# Deployment script for HRL Finance System

set -e  # Exit on error

echo "=========================================="
echo "HRL Finance System - Deployment"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Parse command line arguments
ENVIRONMENT=${1:-production}
ACTION=${2:-deploy}

echo -e "${GREEN}Environment: $ENVIRONMENT${NC}"
echo -e "${GREEN}Action: $ACTION${NC}"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo -e "${RED}Error: .env file not found${NC}"
    echo -e "${YELLOW}Copy .env.example to .env and configure it${NC}"
    exit 1
fi

# Function to check if service is healthy
check_health() {
    local max_attempts=30
    local attempt=1
    
    echo -e "\n${YELLOW}Waiting for service to be healthy...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:${PORT:-8000}/health > /dev/null 2>&1; then
            echo -e "${GREEN}Service is healthy!${NC}"
            return 0
        fi
        echo -e "Attempt $attempt/$max_attempts - Service not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}Service failed to become healthy${NC}"
    return 1
}

# Deploy function
deploy() {
    echo -e "\n${GREEN}Starting deployment...${NC}"
    
    # Pull latest changes (if using git)
    if [ -d .git ]; then
        echo -e "${YELLOW}Pulling latest changes...${NC}"
        git pull
    fi
    
    # Build the application
    echo -e "\n${YELLOW}Building application...${NC}"
    ./scripts/build.sh
    
    # Stop existing containers
    echo -e "\n${YELLOW}Stopping existing containers...${NC}"
    docker-compose down
    
    # Start new containers
    echo -e "\n${YELLOW}Starting new containers...${NC}"
    docker-compose up -d
    
    # Check health
    if check_health; then
        echo -e "\n${GREEN}=========================================="
        echo -e "Deployment completed successfully!"
        echo -e "==========================================${NC}"
        echo -e "\nApplication is running at: http://localhost:${PORT:-8000}"
        echo -e "API documentation: http://localhost:${PORT:-8000}/docs"
    else
        echo -e "\n${RED}Deployment failed - service is not healthy${NC}"
        echo -e "${YELLOW}Check logs with: docker-compose logs${NC}"
        exit 1
    fi
}

# Rollback function
rollback() {
    echo -e "\n${YELLOW}Rolling back to previous version...${NC}"
    
    # Stop current containers
    docker-compose down
    
    # Start with previous image (if tagged)
    if [ ! -z "$PREVIOUS_VERSION" ]; then
        docker tag hrl-finance:$PREVIOUS_VERSION hrl-finance:latest
        docker-compose up -d
        
        if check_health; then
            echo -e "\n${GREEN}Rollback completed successfully!${NC}"
        else
            echo -e "\n${RED}Rollback failed${NC}"
            exit 1
        fi
    else
        echo -e "${RED}No previous version found${NC}"
        exit 1
    fi
}

# Status function
status() {
    echo -e "\n${GREEN}Checking service status...${NC}"
    docker-compose ps
    
    echo -e "\n${GREEN}Checking health endpoint...${NC}"
    if curl -f http://localhost:${PORT:-8000}/health; then
        echo -e "\n${GREEN}Service is healthy${NC}"
    else
        echo -e "\n${RED}Service is not responding${NC}"
    fi
}

# Logs function
logs() {
    echo -e "\n${GREEN}Showing logs...${NC}"
    docker-compose logs -f --tail=100
}

# Execute action
case $ACTION in
    deploy)
        deploy
        ;;
    rollback)
        rollback
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    *)
        echo -e "${RED}Unknown action: $ACTION${NC}"
        echo -e "Usage: $0 [environment] [action]"
        echo -e "  environment: production (default), staging, development"
        echo -e "  action: deploy (default), rollback, status, logs"
        exit 1
        ;;
esac
