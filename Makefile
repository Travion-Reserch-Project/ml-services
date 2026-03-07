# Makefile for Travion ML Services Deployment
# Simplifies common Docker and development tasks

.PHONY: help build up down logs status clean test lint format

# Variables
COMPOSE := docker-compose
COMPOSE_PROD := docker-compose -f docker-compose.production.yml
PYTHON := python3
SERVICES := transport-service safety-service weather-service ai-agent-engine

# Color output
BLUE := \033[0;34m
GREEN := \033[0;32m
RED := \033[0;31m
NC := \033[0m

help:
	@echo "$(BLUE)Travion ML Services - Make Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Build Commands:$(NC)"
	@echo "  make build              Build all Docker images"
	@echo "  make build-service      Build specific service (e.g., make build-service SERVICE=transport)"
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@echo "  make dev                Start all services in development mode"
	@echo "  make up                 Start all services (detached)"
	@echo "  make down               Stop all services"
	@echo "  make restart            Restart all services"
	@echo "  make logs               View logs from all services"
	@echo "  make logs-service       View logs for specific service (make logs SERVICE=transport)"
	@echo ""
	@echo "$(GREEN)Production:$(NC)"
	@echo "  make prod               Build and start production deployment"
	@echo "  make prod-down          Stop production services"
	@echo ""
	@echo "$(GREEN)Testing:$(NC)"
	@echo "  make test               Run all tests"
	@echo "  make test-service       Test specific service (make test-service SERVICE=transport)"
	@echo "  make lint               Run code linting"
	@echo "  make format             Format code"
	@echo ""
	@echo "$(GREEN)Utilities:$(NC)"
	@echo "  make status             Show service status"
	@echo "  make stats              Show resource usage"
	@echo "  make shell              Open shell in service container"
	@echo "  make clean              Clean up containers, volumes, images"
	@echo "  make health             Check all service health endpoints"
	@echo "  make prune              Remove unused Docker resources"
	@echo ""

# ============================================================================
# BUILD COMMANDS
# ============================================================================

build:
	@echo "$(BLUE)Building all Docker images...$(NC)"
	$(COMPOSE) build
	@echo "$(GREEN)✓ Build completed$(NC)"

build-no-cache:
	@echo "$(BLUE)Building all Docker images (no cache)...$(NC)"
	$(COMPOSE) build --no-cache
	@echo "$(GREEN)✓ Build completed$(NC)"

build-service:
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Error: SERVICE not specified$(NC)"; \
		echo "Usage: make build-service SERVICE=transport"; \
		exit 1; \
	fi
	@echo "$(BLUE)Building $(SERVICE)-service...$(NC)"
	$(COMPOSE) build $(SERVICE)-service
	@echo "$(GREEN)✓ Build completed$(NC)"

# ============================================================================
# DEVELOPMENT COMMANDS
# ============================================================================

dev:
	@echo "$(BLUE)Starting services in development mode...$(NC)"
	$(COMPOSE) up
	@echo "$(GREEN)✓ Services started$(NC)"

up:
	@echo "$(BLUE)Starting all services in background...$(NC)"
	$(COMPOSE) up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo "   Transport Service: http://localhost:8001"
	@echo "   Safety Service:    http://localhost:8003"
	@echo "   Weather Service:   http://localhost:8002"
	@echo "   AI Agent Engine:   http://localhost:8004"

down:
	@echo "$(BLUE)Stopping all services...$(NC)"
	$(COMPOSE) down
	@echo "$(GREEN)✓ Services stopped$(NC)"

restart:
	@echo "$(BLUE)Restarting all services...$(NC)"
	$(COMPOSE) restart
	@echo "$(GREEN)✓ Services restarted$(NC)"

# ============================================================================
# LOGGING COMMANDS
# ============================================================================

logs:
	@$(COMPOSE) logs -f

logs-service:
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Error: SERVICE not specified$(NC)"; \
		echo "Usage: make logs-service SERVICE=transport"; \
		exit 1; \
	fi
	@$(COMPOSE) logs -f $(SERVICE)-service

logs-all:
	@$(COMPOSE) logs --tail=100

# ============================================================================
# PRODUCTION COMMANDS
# ============================================================================

prod: build
	@echo "$(BLUE)Starting production deployment...$(NC)"
	$(COMPOSE_PROD) up -d
	@echo "$(GREEN)✓ Production services started$(NC)"

prod-down:
	@echo "$(BLUE)Stopping production services...$(NC)"
	$(COMPOSE_PROD) down
	@echo "$(GREEN)✓ Production services stopped$(NC)"

prod-logs:
	@$(COMPOSE_PROD) logs -f

# ============================================================================
# STATUS & MONITORING
# ============================================================================

status:
	@echo "$(BLUE)Service Status:$(NC)"
	@$(COMPOSE) ps

stats:
	@echo "$(BLUE)Container Statistics:$(NC)"
	@docker stats --no-stream

shell:
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Error: SERVICE not specified$(NC)"; \
		echo "Usage: make shell SERVICE=transport"; \
		exit 1; \
	fi
	@echo "$(BLUE)Opening shell in $(SERVICE)-service...$(NC)"
	@$(COMPOSE) exec $(SERVICE)-service bash

health:
	@echo "$(BLUE)Checking service health...$(NC)"
	@echo "$(BLUE)Transport Service:$(NC)"
	@curl -s http://localhost:8001/api/health || echo "$(RED)Unreachable$(NC)"
	@echo ""
	@echo "$(BLUE)Safety Service:$(NC)"
	@curl -s http://localhost:8003/api/safety/health || echo "$(RED)Unreachable$(NC)"
	@echo ""
	@echo "$(BLUE)Weather Service:$(NC)"
	@curl -s http://localhost:8002/health || echo "$(RED)Unreachable$(NC)"
	@echo ""
	@echo "$(BLUE)AI Agent Engine:$(NC)"
	@curl -s http://localhost:8004/api/v1/health || echo "$(RED)Unreachable$(NC)"

# ============================================================================
# TESTING COMMANDS
# ============================================================================

test:
	@echo "$(BLUE)Running tests...$(NC)"
	@for service in $(SERVICES); do \
		echo "$(BLUE)Testing $$service...$(NC)"; \
		$(COMPOSE) exec -T $$service pytest tests/ || true; \
	done

test-service:
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Error: SERVICE not specified$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Testing $(SERVICE)-service...$(NC)"
	@$(COMPOSE) exec $(SERVICE)-service pytest tests/

lint:
	@echo "$(BLUE)Linting code...$(NC)"
	@$(PYTHON) -m pylint **/*.py --disable=all --enable=E,F || true

format:
	@echo "$(BLUE)Formatting code...$(NC)"
	@$(PYTHON) -m black **/*.py || true

# ============================================================================
# CLEANUP COMMANDS
# ============================================================================

clean:
	@echo "$(BLUE)Cleaning up...$(NC)"
	@$(COMPOSE) down
	@docker system prune -f
	@echo "$(GREEN)✓ Cleanup completed$(NC)"

clean-all:
	@echo "$(RED)WARNING: This will remove all Docker resources!$(NC)"
	@read -p "Continue? (y/N) " confirm && [ "$$confirm" = "y" ] || exit 1
	@$(COMPOSE) down -v
	@docker system prune -af --volumes
	@echo "$(GREEN)✓ Full cleanup completed$(NC)"

prune:
	@echo "$(BLUE)Pruning unused Docker resources...$(NC)"
	@docker container prune -f
	@docker image prune -f
	@docker volume prune -f
	@docker network prune -f
	@echo "$(GREEN)✓ Pruning completed$(NC)"

logs-clean:
	@echo "$(BLUE)Cleaning up container logs...$(NC)"
	@truncate -s 0 /var/lib/docker/containers/*/*-json.log || true
	@echo "$(GREEN)✓ Logs cleaned$(NC)"

# ============================================================================
# DATABASE & BACKUP COMMANDS
# ============================================================================

db-backup:
	@echo "$(BLUE)Creating database backup...$(NC)"
	@mkdir -p ./backups
	@$(COMPOSE) exec -T transport-service python -c \
		"from utils import backup; backup.create_backup('./backups')" || echo "$(RED)Backup command not available$(NC)"
	@echo "$(GREEN)✓ Backup completed$(NC)"

volumes-list:
	@echo "$(BLUE)Docker Volumes:$(NC)"
	@docker volume ls | grep travion

volumes-inspect:
	@echo "$(BLUE)Volume Details:$(NC)"
	@docker volume inspect $$(docker volume ls -q | grep travion) | grep -E '"Name"|"Mountpoint"'

# ============================================================================
# MODEL & DATA COMMANDS
# ============================================================================

models-download:
	@echo "$(BLUE)Downloading ML models...$(NC)"
	@$(COMPOSE) exec -T transport-service python download_model.py || echo "$(RED)Transport model download failed$(NC)"
	@$(COMPOSE) exec -T safety-service python -c "from utils.model_downloader import download_safety_models; download_safety_models()" || echo "$(RED)Safety model download failed$(NC)"
	@echo "$(GREEN)✓ Model download completed$(NC)"

models-list:
	@echo "$(BLUE)Models in Transport Service:$(NC)"
	@$(COMPOSE) exec -T transport-service ls -lh model/ || echo "$(RED)No models found$(NC)"
	@echo ""
	@echo "$(BLUE)Models in Safety Service:$(NC)"
	@$(COMPOSE) exec -T safety-service ls -lh model/ || echo "$(RED)No models found$(NC)"

# ============================================================================
# DEPLOYMENT INFO
# ============================================================================

info:
	@echo "$(BLUE)Deployment Information:$(NC)"
	@echo ""
	@echo "$(GREEN)Images:$(NC)"
	@docker images | grep travion
	@echo ""
	@echo "$(GREEN)Containers:$(NC)"
	@$(COMPOSE) ps
	@echo ""
	@echo "$(GREEN)Networks:$(NC)"
	@docker network ls | grep ml-services

setup:
	@echo "$(BLUE)Setting up project...$(NC)"
	@if [ ! -f ".env" ]; then \
		echo "$(BLUE)Creating .env file...$(NC)"; \
		cp .env.example .env 2>/dev/null || echo "No .env.example found"; \
	fi
	@echo "$(GREEN)✓ Setup completed. Edit .env with your configuration.$(NC)"

# ============================================================================
# QUICK START
# ============================================================================

quickstart: setup build up health
	@echo ""
	@echo "$(GREEN)✓ Quick start completed!$(NC)"
	@echo "$(BLUE)Services are running:$(NC)"
	@make info

# ============================================================================
# DOCKER SYSTEM COMMANDS
# ============================================================================

docker-login:
	@echo "$(BLUE)Logging into Docker registry...$(NC)"
	@docker login

push-images: docker-login
	@echo "$(BLUE)Pushing images to registry...$(NC)"
	@for service in $(SERVICES); do \
		docker push travion/$$service:latest || true; \
	done
	@echo "$(GREEN)✓ Push completed$(NC)"

# ============================================================================
# DEFAULT TARGET
# ============================================================================

.DEFAULT_GOAL := help
