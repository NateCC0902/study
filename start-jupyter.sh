#!/bin/zsh
# Start JupyterLab for the study project (required by the "jupyter" MCP server).
# Run this before study sessions that need notebook execution/plotting.
set -e
source /opt/homebrew/anaconda3/etc/profile.d/conda.sh
conda activate study
TOKEN=$(cat ~/.claude/mcp-servers/jupyter-token)
cd "$(dirname "$0")"
exec jupyter lab --port 8888 --IdentityProvider.token "$TOKEN" --no-browser
