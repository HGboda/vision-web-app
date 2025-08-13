#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Vision Web App Development Starter ===${NC}"
echo -e "${BLUE}This script will start both the Flask backend and React frontend for development.${NC}"
echo

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo -e "${YELLOW}Python is not installed. Please install Python 3.8+ to run the backend.${NC}"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo -e "${YELLOW}npm is not installed. Please install Node.js and npm to run the frontend.${NC}"
    exit 1
fi

# Install backend dependencies
echo -e "${BLUE}Installing backend dependencies...${NC}"
pip install -r requirements.txt

# Install frontend dependencies if node_modules doesn't exist
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${BLUE}Installing frontend dependencies...${NC}"
    cd frontend && npm install && cd ..
fi

# Start the Flask backend in the background (on port 5000 to match React proxy)
echo -e "${BLUE}Starting Flask backend server on port 5000...${NC}"
PORT=5000 python3 api.py &
BACKEND_PID=$!

# Give the backend a moment to start
sleep 2

# Start the React frontend
echo -e "${BLUE}Starting React frontend development server...${NC}"
cd frontend && npm start

# When the React process exits, kill the Flask backend
echo -e "${BLUE}Shutting down Flask backend...${NC}"
kill $BACKEND_PID

echo -e "${GREEN}Development servers stopped.${NC}"
