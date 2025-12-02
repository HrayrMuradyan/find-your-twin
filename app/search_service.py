from pathlib import Path
import zmq
import sys
import logging

# Add the project root to the path
script_dir = Path(__file__).parent
PROJECT_ROOT = script_dir.parent
sys.path.append(str(PROJECT_ROOT))

from src.search_engine import VectorDB
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def run_search_server():
    """
    Main loop for the ZeroMQ Search Service.
    Designed to run as a persistent background process.
    """
    try:
        vectordb = VectorDB()
    except Exception as e:
        logger.critical("Failed to initialize VectorDB: %s", e)
        sys.exit(1)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    
    try:
        socket.bind("tcp://127.0.0.1:5555")
        logger.info("ZeroMQ Search Server listening on tcp://127.0.0.1:5555")
    except zmq.ZMQError as e:
        logger.critical("Failed to bind ZMQ port 5555: %s", e)
        sys.exit(1)

    while True:
        try:
            # Receive the message
            msg = socket.recv_pyobj()

            # Get the command
            cmd = msg.get("command")
            response = {"status": "error", "msg": "Unknown command"}

            if cmd == "search":
                if "vector" in msg:
                    res = vectordb.search(msg["vector"], msg.get("k", 5))
                    response = {"status": "ok", "results": res}
                else:
                    response["msg"] = "Missing 'vector' field"
            
            elif cmd == "add":
                if "id" in msg and "vector" in msg:
                    vectordb.add_item(
                        msg["id"],
                        msg["vector"],
                        msg.get("metadata", {})
                    )
                    response = {"status": "ok"}
                else:
                    response["msg"] = "Missing id or vector for add"
            
            elif cmd == "delete":
                if "id" in msg:
                    vectordb.remove_item(msg["id"])
                    response = {"status": "ok"}
                else:
                    response["msg"] = "Missing id for delete"

            elif cmd == "health":
                response = {
                    "status": "ok",
                    "count": vectordb.index.ntotal if vectordb.index else 0
                }
            
            # Send response
            socket.send_pyobj(response)

        except zmq.ZMQError as ze:
            logger.error("ZMQ Communication Error: %s", ze)
            # In some cases (like EFSM) we might need to reset, 
            # but for REP/REQ usually we just wait for next.
        except Exception as e:
            logger.exception("Unexpected error in search loop: %s", e)
            # Attempt to send error back so client doesn't hang
            try:
                socket.send_pyobj({"status": "error", "msg": str(e)})
            except:
                pass

if __name__ == "__main__":
    run_search_server()