#!/usr/bin/env python3
"""
Command-line utility to shutdown the viz_server.

Usage:
    python3 shutdown_viz_server.py
    or
    ./shutdown_viz_server.py
"""

import sys
import zmq
from termcolor import cprint

def shutdown_server(port: int = 5556) -> bool:
    """Send shutdown command to viz_server."""
    try:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 2000)  # 2 second timeout
        sock.connect(f"tcp://127.0.0.1:{port}")
        
        # Send shutdown command
        sock.send_json({"cmd": "shutdown"})
        resp = sock.recv_json()
        
        sock.close()
        ctx.term()
        
        if isinstance(resp, dict) and resp.get("status") == "ok":
            cprint("✅ viz_server shutdown command sent successfully", "green")
            
            # Wait a moment for cleanup to complete
            import time
            time.sleep(2)
            
            # Verify processes are gone
            return verify_shutdown()
        else:
            msg = resp.get("msg", "unknown error") if isinstance(resp, dict) else str(resp)
            cprint(f"❌ Error from server: {msg}", "red")
            return False
            
    except zmq.Again:
        cprint("❌ Timeout: viz_server not responding (may not be running)", "red")
        return False
    except Exception as e:
        cprint(f"❌ Error connecting to viz_server: {e}", "red")
        return False


def verify_shutdown() -> bool:
    """Verify that viz_server and robot_state_publisher processes are gone."""
    import subprocess
    try:
        # Check for viz_server processes
        result = subprocess.run(
            ["pgrep", "-f", "viz_server.server"], 
            capture_output=True, text=True
        )
        if result.returncode == 0:
            cprint("⚠️  Warning: viz_server processes still running", "yellow")
            return False
            
        # Check for robot_state_publisher processes
        result = subprocess.run(
            ["pgrep", "-f", "robot_state_publisher"], 
            capture_output=True, text=True
        )
        if result.returncode == 0:
            cprint("⚠️  Warning: robot_state_publisher processes still running", "yellow")
            return False
            
        cprint("✅ All processes terminated successfully", "green")
        return True
        
    except Exception as e:
        cprint(f"⚠️  Could not verify process termination: {e}", "yellow")
        return True  # Assume success if we can't verify

def force_kill_processes() -> bool:
    """Force kill any remaining viz_server or robot_state_publisher processes."""
    import subprocess
    import os
    killed_any = False
    
    try:
        # Get our own process ID to avoid killing ourselves
        my_pid = os.getpid()
        
        # Force kill viz_server processes (excluding this script)
        # Use more specific pattern to avoid matching test scripts
        result = subprocess.run(["pgrep", "-f", "viz_server.server"], capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid.strip() and int(pid.strip()) != my_pid:
                    try:
                        subprocess.run(["kill", "-9", pid.strip()], check=True)
                        cprint(f"🔪 Force killed viz_server process {pid.strip()}", "yellow")
                        killed_any = True
                    except subprocess.CalledProcessError:
                        pass  # Process might have already died
            
        # Force kill robot_state_publisher processes  
        result = subprocess.run(["pgrep", "-f", "robot_state_publisher"], capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid.strip():
                    try:
                        subprocess.run(["kill", "-9", pid.strip()], check=True)
                        cprint(f"🔪 Force killed robot_state_publisher process {pid.strip()}", "yellow")
                        killed_any = True
                    except subprocess.CalledProcessError:
                        pass  # Process might have already died
            
        if killed_any:
            cprint("✅ Force kill completed", "green")
        else:
            cprint("ℹ️  No processes to force kill", "cyan")
            
        return True
        
    except Exception as e:
        cprint(f"❌ Error during force kill: {e}", "red")
        return False


def main():
    """Main entry point."""
    force_kill = False
    port = 5556
    
    # Parse arguments
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg in ["-h", "--help"]:
            print(__doc__)
            print("\nOptions:")
            print("  -f, --force    Force kill processes if graceful shutdown fails")
            print("  <port>         Port number (default: 5556)")
            return
        elif arg in ["-f", "--force"]:
            force_kill = True
        else:
            try:
                port = int(arg)
            except ValueError:
                cprint("❌ Invalid port number", "red")
                sys.exit(1)
    
    cprint(f"🛑 Attempting to shutdown viz_server on port {port}...", "yellow")
    
    if shutdown_server(port):
        cprint("🎉 viz_server shutdown completed", "green")
        sys.exit(0)
    else:
        cprint("💥 Graceful shutdown failed", "red")
        
        if force_kill:
            cprint("🔪 Attempting force kill...", "yellow")
            if force_kill_processes():
                cprint("🎉 Force shutdown completed", "green")
                sys.exit(0)
            else:
                cprint("💥 Force kill also failed", "red")
                sys.exit(1)
        else:
            cprint("💡 Try running with --force flag to force kill processes", "cyan")
            sys.exit(1)

if __name__ == "__main__":
    main() 