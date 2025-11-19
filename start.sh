#!/bin/bash

# Ramajan - Mathematical Approximation Evaluator
# Startup script with ASCII vibes

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                              ║"
echo "║    ██████╗  █████╗ ███╗   ███╗ █████╗      ██╗ █████╗ ███╗   ██╗           ║"
echo "║    ██╔══██╗██╔══██╗████╗ ████║██╔══██╗     ██║██╔══██╗████╗  ██║           ║"
echo "║    ██████╔╝███████║██╔████╔██║███████║     ██║███████║██╔██╗ ██║           ║"
echo "║    ██╔══██╗██╔══██║██║╚██╔╝██║██╔══██║██   ██║██╔══██║██║╚██╗██║           ║"
echo "║    ██║  ██║██║  ██║██║ ╚═╝ ██║██║  ██║╚█████╔╝██║  ██║██║ ╚████║           ║"
echo "║    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝           ║"
echo "║                                                                              ║"
echo "║    Mathematical Approximation Evaluator                                     ║"
echo "║    Starting up with ASCII vibes ⚡                                          ║"
echo "║                                                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ first."
    echo "   Visit: https://nodejs.org/"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

echo "✅ Node.js version: $(node --version)"
echo "✅ npm version: $(npm --version)"
echo ""

# Check if dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing backend dependencies..."
    npm install
    echo ""
fi

if [ ! -d "client/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd client && npm install && cd ..
    echo ""
fi

echo "🚀 Starting Ramajan..."
echo "┌─ Backend API: http://localhost:5000"
echo "├─ Frontend: http://localhost:3000"
echo "└─ Press Ctrl+C to stop"
echo ""

# Start both backend and frontend
npm run dev
