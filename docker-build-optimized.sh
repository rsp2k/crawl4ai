#!/bin/bash

# =====================================
# OPTIMIZED DOCKER BUILD SCRIPT
# =====================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
INSTALL_TYPE=${INSTALL_TYPE:-"default"}
ENABLE_GPU=${ENABLE_GPU:-"false"}
BUILD_CACHE=${BUILD_CACHE:-"true"}
TAG=${TAG:-"crawl4ai:optimized"}

echo -e "${BLUE}🚀 Crawl4AI Optimized Docker Build${NC}"
echo -e "${BLUE}===================================${NC}"

# Check if BuildKit is available
if ! docker buildx version >/dev/null 2>&1; then
    echo -e "${RED}❌ Docker BuildKit is required for optimization features!${NC}"
    echo -e "${YELLOW}Please update Docker or enable BuildKit${NC}"
    exit 1
fi

# Backup original files if they exist
if [ -f "Dockerfile" ] && [ ! -f "Dockerfile.original" ]; then
    echo -e "${YELLOW}📦 Backing up original Dockerfile...${NC}"
    cp Dockerfile Dockerfile.original
fi

if [ -f ".dockerignore" ] && [ ! -f ".dockerignore.original" ]; then
    echo -e "${YELLOW}📦 Backing up original .dockerignore...${NC}"
    cp .dockerignore .dockerignore.original
fi

# Use optimized files
echo -e "${GREEN}✨ Using optimized build configuration...${NC}"
cp Dockerfile.optimized Dockerfile
cp .dockerignore.optimized .dockerignore

# Build arguments
BUILD_ARGS=(
    "--build-arg" "INSTALL_TYPE=${INSTALL_TYPE}"
    "--build-arg" "ENABLE_GPU=${ENABLE_GPU}"
    "--build-arg" "TARGETARCH=$(docker version --format '{{.Server.Arch}}')"
)

# Cache options
if [ "$BUILD_CACHE" = "true" ]; then
    echo -e "${GREEN}🗄️  Using build cache for faster builds...${NC}"
    BUILD_ARGS+=("--cache-from" "type=local,src=/tmp/.buildx-cache")
    BUILD_ARGS+=("--cache-to" "type=local,dest=/tmp/.buildx-cache")
fi

# Progress output
BUILD_ARGS+=("--progress" "plain")

echo -e "${BLUE}📋 Build Configuration:${NC}"
echo -e "  Install Type: ${YELLOW}${INSTALL_TYPE}${NC}"
echo -e "  GPU Support: ${YELLOW}${ENABLE_GPU}${NC}"
echo -e "  Build Cache: ${YELLOW}${BUILD_CACHE}${NC}"
echo -e "  Tag: ${YELLOW}${TAG}${NC}"
echo ""

# Estimate build time
echo -e "${BLUE}⏱️  Estimated Build Times:${NC}"
if [ "$BUILD_CACHE" = "true" ]; then
    echo -e "  First build: ${YELLOW}15-25 minutes${NC} (downloading AI/ML packages)"
    echo -e "  Code changes: ${GREEN}2-5 minutes${NC} (cached dependencies)"
    echo -e "  No changes: ${GREEN}30-60 seconds${NC} (cached everything)"
else
    echo -e "  Every build: ${RED}15-25 minutes${NC} (no caching)"
fi
echo ""

# Start build
echo -e "${GREEN}🔨 Starting optimized build...${NC}"
echo -e "${BLUE}This may take a while on first build (downloading AI/ML packages)${NC}"
echo ""

# Build with timing
start_time=$(date +%s)

if docker buildx build \
    "${BUILD_ARGS[@]}" \
    --tag "${TAG}" \
    --load \
    . ; then
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    minutes=$((duration / 60))
    seconds=$((duration % 60))
    
    echo ""
    echo -e "${GREEN}✅ Build completed successfully!${NC}"
    echo -e "${BLUE}📊 Build time: ${minutes}m ${seconds}s${NC}"
    echo -e "${BLUE}🏷️  Image tagged as: ${TAG}${NC}"
    
    # Show image size
    size=$(docker images "${TAG}" --format "table {{.Size}}" | tail -n 1)
    echo -e "${BLUE}💾 Image size: ${size}${NC}"
    
    echo ""
    echo -e "${GREEN}🚀 Ready to run:${NC}"
    echo -e "  ${BLUE}docker run -d -p 11235:11235 ${TAG}${NC}"
    
else
    echo -e "${RED}❌ Build failed!${NC}"
    
    # Restore original files on failure
    if [ -f "Dockerfile.original" ]; then
        echo -e "${YELLOW}🔄 Restoring original Dockerfile...${NC}"
        mv Dockerfile.original Dockerfile
    fi
    
    if [ -f ".dockerignore.original" ]; then
        echo -e "${YELLOW}🔄 Restoring original .dockerignore...${NC}"
        mv .dockerignore.original .dockerignore
    fi
    
    exit 1
fi

# Clean up
echo ""
echo -e "${BLUE}🧹 Cleaning up...${NC}"

# Restore original files
if [ -f "Dockerfile.original" ]; then
    mv Dockerfile.original Dockerfile
fi

if [ -f ".dockerignore.original" ]; then
    mv .dockerignore.original .dockerignore
fi

echo -e "${GREEN}✨ Optimization complete!${NC}"