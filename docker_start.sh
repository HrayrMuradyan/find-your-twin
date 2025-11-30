#!/bin/bash
set -e

# Setup Credentials
echo "-----------------------------------"
echo "Setting up Credentials..."
mkdir -p credentials

if [ -n "$GOOGLE_CLIENT_SECRET" ]; then 
    echo "$GOOGLE_CLIENT_SECRET" > credentials/client_secret.json
    echo "   - client_secret.json created"
fi

if [ -n "$GOOGLE_TOKEN_JSON" ]; then 
    echo "$GOOGLE_TOKEN_JSON" > credentials/token.json
    echo "   - token.json created"
fi

# Start the supervisord
echo "Starting Supervisord..."
exec supervisord -c app/supervisord.conf