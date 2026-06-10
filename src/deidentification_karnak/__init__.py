# Load environment variables from a .env file before any submodule reads them.
# Submodules (e.g. image_processing) read os.environ at import time, so this
# must run first.
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True), override=False)
