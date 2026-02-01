#!/bin/bash
# Alpha Platform - Directory Structure Setup

echo "Creating Alpha Platform directory structure..."

# Root level directories
mkdir -p config
mkdir -p sql
mkdir -p scripts
mkdir -p workers
mkdir -p tests
mkdir -p logs
mkdir -p data/cache

# Source code directories
mkdir -p src/db
mkdir -p src/kafka
mkdir -p src/data
mkdir -p src/screener
mkdir -p src/committee/agents
mkdir -p src/llm
mkdir -p src/portfolio
mkdir -p src/utils

# Dashboard directories
mkdir -p dashboard/pages
mkdir -p dashboard/components
mkdir -p dashboard/styles

echo "Directory structure created successfully!"
echo ""
echo "Structure:"
find . -type d | head -40
