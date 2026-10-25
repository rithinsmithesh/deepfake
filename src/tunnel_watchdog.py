#!/usr/bin/env python3
"""
DeepShield Resilient Watchdog & Tunnel Manager
Ensures Streamlit is running, maintains Cloudflare tunnel,
pings to keep connection active, and updates the Desktop link files.
"""

import os
import sys
import time
import subprocess
import re
import socket

WORKSPACE_DIR = "/Users/hemanshmurahari/Desktop/DL Project"
DESKTOP_DIR = "/Users/hemanshmurahari/Desktop"
LINK_FILE = os.path.join(DESKTOP_DIR, "DEEPSHIELD_CURRENT_LINK.txt")
HTML_FILE = os.path.join(DESKTOP_DIR, "Open_DeepShield.html")
CLOUDFLARED_BIN = "/tmp/cloudflared"
PORT = 8501

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def is_port_open(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(1.5)
        res = s.connect_ex(('127.0.0.1', port))
        return res == 0
    except Exception:
        return False
    finally:
        s.close()

def ensure_streamlit():
    if not is_port_open(PORT):
        log(f"Streamlit not detected on port {PORT}. Launching...")
        cmd = f"nohup python3 -m streamlit run app.py --server.port {PORT} --server.headless true --server.enableCORS false --server.enableXsrfProtection false > /tmp/streamlit_app.log 2>&1 &"
        subprocess.Popen(cmd, shell=True, cwd=WORKSPACE_DIR)
        for _ in range(15):
            time.sleep(1)
            if is_port_open(PORT):
                log("Streamlit started successfully.")
                return True
        log("Warning: Streamlit start timed out.")
        return False
    return True

def get_local_ip():
    try:
        res = subprocess.check_output(
            "ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || ifconfig | grep 'inet ' | grep -v 127.0.0.1 | head -n1 | awk '{print $2}'",
            shell=True
        )
        ip = res.decode().strip()
        if ip:
            return ip
    except Exception:
        pass
    return "127.0.0.1"

def update_desktop_files(tunnel_url):
    local_ip = get_local_ip()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    content = f"""===================================================================
DEEPSHIELD FORENSIC SUITE - OFFICIAL ACCESS LINKS
Updated: {timestamp}
===================================================================

1. 💻 ON THIS MAC (BEST FOR DEMOS & REVIEWS - 0ms latency, NEVER EXPIRES):
   ▶ http://localhost:{PORT}

2. 📱 ON SAME WIFI / PHONE HOTSPOT (For reviewers on same network):
   ▶ http://{local_ip}:{PORT}

3. 🌐 PUBLIC WORLDWIDE LINK (For Windows, Mac, Android, iPhone anywhere):
   ▶ {tunnel_url}

RECOMMENDATION:
- For presenting from this Mac: ALWAYS use http://localhost:{PORT}
  (100% reliable, zero cloud lag, immune to internet disconnections)
- For external/remote access: Use the public link above.
===================================================================
"""
    try:
        with open(LINK_FILE, "w") as f:
            f.write(content)
    except Exception as e:
        log(f"Failed to write {LINK_FILE}: {e}")

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepShield Forensic Suite</title>
    <style>
        body {{
            background: #090d16;
            color: #f8fafc;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
            box-sizing: border-box;
        }}
        .card {{
            background: rgba(15, 23, 42, 0.95);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 16px;
            padding: 36px;
            max-width: 520px;
            width: 100%;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
            text-align: center;
            box-sizing: border-box;
        }}
        h1 {{
            color: #38bdf8;
            font-size: 1.6rem;
            margin-bottom: 8px;
        }}
        p {{
            color: #94a3b8;
            font-size: 0.95rem;
            margin-bottom: 24px;
        }}
        .btn {{
            display: block;
            width: 100%;
            padding: 14px 20px;
            margin-bottom: 12px;
            border-radius: 10px;
            text-decoration: none;
            font-weight: 700;
            font-size: 1rem;
            box-sizing: border-box;
            transition: all 0.2s;
        }}
        .btn-local {{
            background: linear-gradient(135deg, #0ea5e9, #6366f1);
            color: #ffffff;
        }}
        .btn-public {{
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #38bdf8;
        }}
        .btn:hover {{
            transform: translateY(-2px);
            opacity: 0.95;
        }}
        .badge {{
            display: inline-block;
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.4);
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-family: monospace;
            margin-bottom: 16px;
        }}
    </style>
</head>
<body>
    <div class="card">
        <span class="badge">● SYSTEM OPERATIONAL</span>
        <h1>🛡️ DeepShield Forensics</h1>
        <p>Choose your access mode:</p>
        
        <a class="btn btn-local" href="http://localhost:{PORT}">
            💻 Open Local App (0ms Latency - Recommended)
        </a>
        
        <a class="btn btn-public" href="{tunnel_url}" target="_blank">
            🌐 Open Public Cloud Link (Any Device / OS)
        </a>
        
        <div style="margin-top: 18px; font-size: 0.8rem; color: #64748b;">
            Updated: {timestamp}
        </div>
    </div>
</body>
</html>
"""
    try:
        with open(HTML_FILE, "w") as f:
            f.write(html_content)
    except Exception as e:
        log(f"Failed to write {HTML_FILE}: {e}")

def run_tunnel():
    ensure_streamlit()

    subprocess.call("pkill -f 'cloudflared tunnel' || true", shell=True)
    time.sleep(1)

    log_file = "/tmp/cloudflared_watchdog.log"
    if os.path.exists(log_file):
        os.remove(log_file)

    cmd = [
        CLOUDFLARED_BIN,
        "tunnel",
        "--edge-ip-version", "4",
        "--protocol", "http2",
        "--url", f"http://127.0.0.1:{PORT}"
    ]

    log("Launching cloudflared tunnel...")
    with open(log_file, "w") as out_f:
        proc = subprocess.Popen(cmd, stdout=out_f, stderr=out_f)

    tunnel_url = None
    for _ in range(30):
        time.sleep(0.5)
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                content = f.read()
                m = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", content)
                if m:
                    tunnel_url = m.group(0)
                    break
        if proc.poll() is not None:
            log("cloudflared exited unexpectedly.")
            return

    if not tunnel_url:
        log("Failed to obtain tunnel URL from logs.")
        if proc.poll() is None:
            proc.terminate()
        return

    log(f"Tunnel Established: {tunnel_url}")
    update_desktop_files(tunnel_url)

    # Monitor loop: gentle keepalive, only restart if process dies or explicit permanent error
    consecutive_errors = 0
    while True:
        time.sleep(45)

        # 1. Process alive check
        if proc.poll() is not None:
            log("Tunnel process stopped. Restarting...")
            break

        # 2. Check for explicit permanent auth failure in log
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    lines = f.readlines()
                    last_5 = "".join(lines[-5:]) if len(lines) >= 5 else "".join(lines)
                    if "Unauthorized: Tunnel not found" in last_5:
                        log("Permanent 'Unauthorized: Tunnel not found' detected. Recreating tunnel...")
                        proc.terminate()
                        break
            except Exception:
                pass

        # 3. Soft keepalive ping (via curl with 5s timeout, quiet)
        ret = subprocess.call(f"curl -s -o /dev/null -m 5 '{tunnel_url}'", shell=True)
        if ret != 0:
            consecutive_errors += 1
            log(f"Keepalive ping returned non-zero ({consecutive_errors}/5)")
        else:
            consecutive_errors = 0

        # Only restart after 5 consecutive failures across 4 minutes
        if consecutive_errors >= 5:
            log("5 consecutive keepalive failures (tunnel appears frozen). Restarting...")
            proc.terminate()
            break

        ensure_streamlit()

def main():
    log("Starting DeepShield Tunnel Supervisor...")
    while True:
        try:
            run_tunnel()
        except Exception as e:
            log(f"Supervisor error: {e}")
        time.sleep(3)

if __name__ == "__main__":
    main()
