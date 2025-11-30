from pathlib import Path
import zmq

# Add the project root to the path
import sys
script_dir = Path(__file__).parent
PROJECT_ROOT = script_dir.parent
sys.path.append(str(PROJECT_ROOT))

from src.search_engine import VectorDB

# Setup logging
import logging
from src.logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

def run_search_server():
    vectordb = VectorDB()
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://127.0.0.1:5555")

    logger.info("ZeroMQ Server listening on port 5555")

    while True:
        try:
            # Receive the message
            msg = socket.recv_pyobj()

            # Get the command
            # There are four commands:
            # "search", "add", "delete", "health"
            cmd = msg.get("command")

            if cmd == "search":
                print("SEARCHIIIING")
                res = vectordb.search(msg["vector"], msg.get("k", 5))
                socket.send_pyobj({"status": "ok", "results": res})
                print("SEARCHIIIING DONE")
            
            elif cmd == "add":
                vectordb.add_item(
                    msg["id"],
                    msg["vector"],
                    msg["metadata"]
                )
                socket.send_pyobj({"status": "ok"})
            
            elif cmd == "delete":
                vectordb.remove_item(msg["id"])
                socket.send_pyobj({"status": "ok"})

            elif cmd == "health":
                socket.send_pyobj({
                    "status": "ok",
                    "count": vectordb.index.ntotal
                })
            
            else:
                socket.send_pyobj({
                    "status": "error",
                    "msg": "Unknown command"
                })
        except Exception as e:
            logger.exception("Error in loop: %s", e)
            socket.send_pyobj({"status": "error", "msg": str(e)})

if __name__ == "__main__":
    run_search_server()


            