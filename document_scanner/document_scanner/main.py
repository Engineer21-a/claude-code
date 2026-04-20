"""Entry point for Document Scanner application."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from document_scanner.ui.main_window import MainWindow


def main(args: Optional[List[str]] = None) -> int:
    app = QApplication(args or sys.argv)
    app.setApplicationName("Document Scanner")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("DocumentScanner")
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)

    window = MainWindow()
    window.show()

    # Accept file paths as CLI arguments
    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:] if Path(p).exists()]
        if paths:
            window.load_images(paths)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
