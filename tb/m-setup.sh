#!/bin/sh
# Configure m inside a Terminal-Bench task container. The adapter copies the
# static musl binary to /usr/local/bin/m before this runs (task images have
# no curl/wget); a python3 download from the release is the standalone
# fallback. M_SERVER_URL is exported by the adapter; if absent, fall back to
# the container's default gateway (the docker host).
set -e

M_RELEASE_URL="https://github.com/dairycow/m/releases/download/v0.1.0/m-x86_64-linux-musl"

if [ ! -x /usr/local/bin/m ]; then
    if [ -f /usr/local/bin/m ]; then
        chmod +x /usr/local/bin/m
    elif command -v python3 >/dev/null 2>&1; then
        python3 -c "import urllib.request; urllib.request.urlretrieve('$M_RELEASE_URL', '/usr/local/bin/m')"
        chmod +x /usr/local/bin/m
    else
        echo "no m binary and no way to download one" >&2
        exit 1
    fi
fi

if [ -z "$M_SERVER_URL" ]; then
    GW=$(ip route 2>/dev/null | awk '/default/ {print $3; exit}')
    M_SERVER_URL="http://${GW:-172.17.0.1}:8080"
fi

CFG_DIR="${HOME:-/root}/.config/m"
mkdir -p "$CFG_DIR"
cat > "$CFG_DIR/config.toml" <<EOF
default_profile = "local"

[profiles.local]
base_url = "$M_SERVER_URL"
api_key = "none"
model = "local"
EOF

m --version
