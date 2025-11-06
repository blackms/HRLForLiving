#!/bin/bash
# Production build script for HRL Finance System

set -e  # Exit on error

echo "=========================================="
echo "HRL Finance System - Production Build"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    exit 1
fi

# Load environment variables if .env exists
if [ -f .env ]; then
    echo -e "${GREEN}Loading environment variables from .env${NC}"
    export $(cat .env | grep -v '^#' | xargs)
else
    echo -e "${YELLOW}Warning: .env file not found, using defaults${NC}"
    echo -e "${YELLOW}Copy .env.example to .env and customize if needed${NC}"
fi

# Clean previous builds
echo -e "\n${YELLOW}Cleaning previous builds...${NC}"
rm -rf frontend/dist
rm -rf backend/__pycache__
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Build Docker image
echo -e "\n${GREEN}Building Docker image...${NC}"
docker build -t hrl-finance:latest -f Dockerfile .

# Tag with version if VERSION is set
if [ ! -z "$VERSION" ]; then
    echo -e "${GREEN}Tagging image with version: $VERSION${NC}"
    docker tag hrl-finance:latest hrl-finance:$VERSION
fi

echo -e "\n${GREEN}=========================================="
echo -e "Build completed successfully!"
echo -e "==========================================${NC}"
echo -e "\nTo run the application:"
echo -e "  ${YELLOW}docker-compose up -d${NC}"
echo -e "\nTo view logs:"
echo -e "  ${YELLOW}docker-compose logs -f${NC}"
echo -e "\nTo stop the application:"
echo -e "  ${YELLOW}docker-compose down${NC}"
