#!/bin/bash
# DataSpur Deployment Script
# Packages backend and frontend for deployment

set -e

echo "========================================="
echo "DataSpur — Deployment Package Builder"
echo "========================================="

DEPLOY_DIR="$HOME/dataspur-deploy"
rm -rf "$DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR/backend/data" "$DEPLOY_DIR/frontend"

# Backend
echo "[1/4] Packaging backend..."
cp backend/main.py "$DEPLOY_DIR/backend/"
# Fix data path for deployment (make it relative)
python3 -c "
import re
path = '$DEPLOY_DIR/backend/main.py'
with open(path) as f: content = f.read()
content = content.replace('Path.home() / \"dataspur\" / \"data\"', 'Path(__file__).parent / \"data\"')
with open(path, 'w') as f: f.write(content)
print('  Path fixed for deployment')
"
cp backend/requirements.txt "$DEPLOY_DIR/backend/"
cp backend/Procfile "$DEPLOY_DIR/backend/"
cp backend/Dockerfile "$DEPLOY_DIR/backend/"
cp data/rides.csv data/bull_profiles.csv data/rider_profiles.csv data/rider_weights.csv "$DEPLOY_DIR/backend/data/" 2>/dev/null

# Frontend
echo "[2/4] Packaging frontend spec..."
cp frontend/COMPONENT_SPEC.md "$DEPLOY_DIR/frontend/"
cp frontend/index.html "$DEPLOY_DIR/frontend/"
cp frontend/package.json "$DEPLOY_DIR/frontend/"
cp DATASPUR_SPEC.md "$DEPLOY_DIR/"

# Documentation
echo "[3/4] Copying documentation..."
cp pbr_2026_research_report.md "$DEPLOY_DIR/" 2>/dev/null

# Summary
echo "[4/4] Done."
echo ""
echo "Package ready at: $DEPLOY_DIR"
echo ""
echo "Structure:"
find "$DEPLOY_DIR" -type f | sort
echo ""
echo "Total size: $(du -sh "$DEPLOY_DIR" | cut -f1)"
echo ""
echo "NEXT STEPS:"
echo "  1. Deploy backend to Railway/Render:"
echo "     cd $DEPLOY_DIR/backend && railway up"
echo ""
echo "  2. Build frontend in lovable.dev:"
echo "     Paste $DEPLOY_DIR/DATASPUR_SPEC.md into lovable.dev prompt"
echo "     Set VITE_API_URL to your Railway URL"
echo ""
echo "  3. Frontend spec at: $DEPLOY_DIR/frontend/COMPONENT_SPEC.md"