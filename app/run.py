import os
import signal
import sys
import time
import logging
import multiprocessing
import uvicorn
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# Import local modules
from src.credentials import setup_google_credentials
from app.search_service import run_search_server 
from src.logging_config import setup_logging

# Initialize Logging
setup_logging()
logger = logging.getLogger("Entrypoint")

# Global process reference for cleanup
search_process = None

def signal_handler(sig, frame):
    """
    Handle Docker/System signals.
    """
    logger.info("Received termination signal (%s). Shutting down...", sig)
    shutdown()
    sys.exit(0)

def shutdown():
    """Terminates the background search process."""
    global search_process
    if search_process and search_process.is_alive():
        logger.info("Terminating Search Service...")
        try:
            # Send SIGTERM to the child
            search_process.terminate()
            # Wait briefly for it to die
            search_process.join(timeout=2)
            
            # If it's still alive (stuck), force kill it
            if search_process.is_alive():
                logger.warning("Search Service did not exit gracefully. Force killing...")
                search_process.kill()
                search_process.join()
                
        except Exception as e:
            logger.error("Error during search process termination: %s", e)
        logger.info("Search Service terminated.")

def start_search_service_wrapper():
    """Wrapper to run the search loop in a separate process"""
    # FIX: Only ignore CTRL+C (SIGINT). 
    # Do NOT ignore SIGTERM, otherwise parent cannot stop this process!
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    try:
        run_search_server()
    except Exception as e:
        logger.critical("Search Service Crashed: %s", e)
        sys.exit(1)

def main():
    global search_process

    # 1. Setup Credentials
    logger.info("Setting up Google Credentials...")
    setup_google_credentials(PROJECT_ROOT.parent)

    # 2. Start Search Service (Background Process)
    logger.info("Starting Vector Search Service (ZeroMQ)...")
    search_process = multiprocessing.Process(target=start_search_service_wrapper, daemon=True)
    search_process.start()

    # 3. Wait briefly for ZMQ to bind
    time.sleep(2) 

    if not search_process.is_alive():
        logger.error("Search Service failed to start immediately. Check logs.")
        sys.exit(1)

    # 4. Start FastAPI (Main Process)
    logger.info("Starting Uvicorn Server on port 7860...")
    try:
        uvicorn.run(
            "app.main:app", 
            host="0.0.0.0", 
            port=7860, 
            workers=2, 
            log_level="info",
            proxy_headers=True,
            forwarded_allow_ips="*"
        )
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception("Uvicorn server crashed: %s", e)
    finally:
        # This block runs when Uvicorn exits (gracefully or via CTRL+C)
        shutdown()

if __name__ == "__main__":
    # Register signal handlers for Docker stops
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    main()