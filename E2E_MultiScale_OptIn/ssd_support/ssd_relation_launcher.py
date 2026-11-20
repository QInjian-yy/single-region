"""Independent SSD entry; never replaces engine.load_images or the legacy launcher."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from E2E_MultiScale_OptIn.multiscale_relation_train import main


if __name__ == "__main__":
    main(default_ssd_root=Path("/home/ub/wsi_fastcache"))
