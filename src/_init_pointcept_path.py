import sys
import os
import logging

# --- ANSI Color Codes for Terminal Output ---
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

# --- Logging Configuration ---
# Format: [Timestamp] [Level] Message
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PointceptWrapper")

# --- 1. Path Configuration ---
try:
    # Get current file directory (src/) and project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Define the path to the Pointcept library (extern/pointcept)
    pointcept_path = os.path.join(project_root, 'extern', 'pointcept')

    # Add the library path to system path if it doesn't exist
    if pointcept_path not in sys.path:
        sys.path.append(pointcept_path)
        logger.info(f"Library path added: {pointcept_path}")
    else:
        logger.debug(f"Library path already exists: {pointcept_path}")

except Exception as e:
    logger.critical(f"{RED}Failed to configure library path: {e}{RESET}")
    sys.exit(1)  # Terminate execution on critical failure

# --- 2. Import Pointcept Modules (Re-export) ---
try:
    # Attempt to import key modules from Pointcept
    # This verifies that the library is correctly linked
    from pointcept.models import build_model
    from pointcept.datasets import build_dataset
    
    # If successful, log a success message in green
    logger.info(f"{GREEN}Pointcept module loaded successfully.{RESET}")

except ImportError as e:
    # Log error if dependencies are missing or path is incorrect
    logger.error(f"{RED}Failed to load Pointcept module.{RESET}")
    logger.error(f"Error Details: {e}")