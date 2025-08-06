#!/bin/bash

# WebGL Build Script for Iron Man Suit Experience
# This script configures and builds the Unity project for WebGL deployment

echo "Iron Man Suit Experience - WebGL Build Script"
echo "============================================"

# Check if Unity is available
UNITY_PATH="/Applications/Unity/Hub/Editor/2021.3.16f1/Unity.app/Contents/MacOS/Unity"
if [ ! -f "$UNITY_PATH" ]; then
    echo "Error: Unity not found at expected path. Please update UNITY_PATH in this script."
    exit 1
fi

# Project path
PROJECT_PATH="../IronManSuitSim3D"
BUILD_PATH="./IronManExperience/Build"

# Create build directory
mkdir -p "$BUILD_PATH"

echo "Building Unity project for WebGL..."

# Unity build command
"$UNITY_PATH" \
    -batchmode \
    -quit \
    -projectPath "$PROJECT_PATH" \
    -buildTarget WebGL \
    -executeMethod WebGLBuilder.BuildProject \
    -logFile build.log

# Check if build succeeded
if [ $? -eq 0 ]; then
    echo "Build completed successfully!"
    echo "Output location: $BUILD_PATH"
    
    # Copy custom HTML template
    if [ -f "webgl-template/index.html" ]; then
        cp webgl-template/index.html "$BUILD_PATH/index.html"
        echo "Custom HTML template applied."
    fi
    
    # Optimize build
    echo "Optimizing build files..."
    
    # Compress with gzip if available
    if command -v gzip &> /dev/null; then
        find "$BUILD_PATH" -name "*.js" -o -name "*.wasm" -o -name "*.data" | while read file; do
            gzip -9 -k "$file"
        done
        echo "Files compressed with gzip."
    fi
    
    echo ""
    echo "Build ready for deployment!"
    echo "To test locally, run: npm run serve"
    
else
    echo "Build failed! Check build.log for details."
    exit 1
fi