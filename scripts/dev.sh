#!/bin/bash
# Development environment startup script

set -e  # Exit on error

echo "=========================================="
echo "HRL Finance System - Development Mode"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Load environment variables if .env exists
if [ -f .env ]; then
    echo -e "${GREEN}Loading environment variables from .env${NC}"
    export $(cat .env | grep -v '^#' | xargs)
else
    echo -e "${YELLOW}Warning: .env file not found${NC}"
    echo -e "${YELLOW}Copy .env.example to .env and customize if needed${NC}"
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    exit 1
fi

# Parse command line arguments
ACTION=${1:-start}

case $ACTION in
    start)
        echo -e "\n${GREEN}Starting development environment...${NC}"
        docker-compose --profile dev up -d hrl-finance-dev
        
        echo -e "\n${GREEN}Development environment started!${NC}"
        echo -e "\nBackend API: http://localhost:${BACKEND_PORT:-8000}"
        echo -e "API Docs: http://localhost:${BACKEND_PORT:-8000}/docs"
        echo -e "Frontend: http://localhost:${FRONTEND_PORT:-5173}"
        echo -e "\nTo view logs:"
        echo -e "  ${YELLOW}./scripts/dev.sh logs${NC}"
        echo -e "\nTo stop:"
        echo -e "  ${YELLOW}./scripts/dev.sh stop${NC}"
        ;;
        
    stop)
        echo -e "\n${YELLOW}Stopping development environment...${NC}"
        docker-compose --profile dev down
        echo -e "${GREEN}Development environment stopped${NC}"
        ;;
        
    restart)
        echo -e "\n${YELLOW}Restarting development environment...${NC}"
        docker-compose --profile dev restart hrl-finance-dev
        echo -e "${GREEN}Development environment restarted${NC}"
        ;;
        
    logs)
        echo -e "\n${GREEN}Showing logs (Ctrl+C to exit)...${NC}"
        docker-compose --profile dev logs -f hrl-finance-dev
        ;;
        
    shell)
        echo -e "\n${GREEN}Opening shell in development container...${NC}"
        docker-compose --profile dev exec hrl-finance-dev /bin/bash
        ;;
        
    backend)
        echo -e "\n${GREEN}Opening backend shell...${NC}"
        docker-compose --profile dev exec hrl-finance-dev /bin/bash -c "cd /app/backend && /bin/bash"
        ;;
        
    frontend)
        echo -e "\n${GREEN}Opening frontend shell...${NC}"
        docker-compose --profile dev exec hrl-finance-dev /bin/bash -c "cd /app/frontend && /bin/bash"
        ;;
        
    test)
        echo -e "\n${GREEN}Running tests...${NC}"
        echo -e "${YELLOW}Backend tests:${NC}"
        docker-compose --profile dev exec hrl-finance-dev /bin/bash -c "cd /app/backend && pytest"
        echo -e "\n${YELLOW}Frontend tests:${NC}"
        docker-compose --profile dev exec hrl-finance-dev /bin/bash -c "cd /app/frontend && npm test"
        ;;
        
    clean)
        echo -e "\n${YELLOW}Cleaning development environment...${NC}"
        docker-compose --profile dev down -v
        docker system prune -f
        echo -e "${GREEN}Development environment cleaned${NC}"
        ;;
        
    *)
        echo -e "${RED}Unknown action: $ACTION${NC}"
        echo -e "\nUsage: $0 [action]"
        echo -e "\nActions:"
        echo -e "  ${GREEN}start${NC}    - Start development environment (default)"
        echo -e "  ${GREEN}stop${NC}     - Stop development environment"
        echo -e "  ${GREEN}restart${NC}  - Restart development environment"
        echo -e "  ${GREEN}logs${NC}     - Show logs"
        echo -e "  ${GREEN}shell${NC}    - Open shell in container"
        echo -e "  ${GREEN}backend${NC}  - Open backend shell"
        echo -e "  ${GREEN}frontend${NC} - Open frontend shell"
        echo -e "  ${GREEN}test${NC}     - Run tests"
        echo -e "  ${GREEN}clean${NC}    - Clean up containers and volumes"
        exit 1
        ;;
esac
