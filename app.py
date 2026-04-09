from __future__ import annotations

import os
import sys


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(REPO_ROOT, "stock_rl_project")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from web_app import app  # noqa: E402


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=False, use_reloader=False, host="0.0.0.0", port=port)
