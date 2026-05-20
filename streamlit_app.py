import os
import sys


APP_DIR = os.path.join(os.path.dirname(__file__), "mindcare_ai_project_content_website")
os.chdir(APP_DIR)
sys.path.insert(0, APP_DIR)

import app  # noqa: E402,F401
