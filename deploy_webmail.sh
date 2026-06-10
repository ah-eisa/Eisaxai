#!/usr/bin/env bash
# deploy_webmail.sh — Build webmail and deploy to eisax-ui/webmail/
set -euo pipefail

WEBMAIL_DIR="/home/ubuntu/eisax-webmail"
DEPLOY_DIR="/home/ubuntu/eisax-ui/webmail"

echo "🔨 Building webmail..."
cd "$WEBMAIL_DIR"
npm run build

echo "🚀 Deploying to $DEPLOY_DIR..."
rm -rf "$DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"
cp -r "$WEBMAIL_DIR/dist/." "$DEPLOY_DIR/"

echo "✅ Webmail deployed — $(ls $DEPLOY_DIR | wc -l) files"
echo "📁 Contents: $(ls $DEPLOY_DIR)"
