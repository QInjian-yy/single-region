"""Evaluate a saved validation split or the official Native512 test set.

Pass official-test as the first argument to use the reference CSV evaluator.
"""

import sys

import eval_e2e
import eval_native512_reference
from models.shared_resnet_sr import CHECKPOINT_FORMAT
from models.shared_resnet_sr_asrd import ASRD_CHECKPOINT_FORMAT, model_from_checkpoint


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "official-test":
        eval_native512_reference.main(
            argv[1:], model_loader=model_from_checkpoint,
            checkpoint_formats=(CHECKPOINT_FORMAT, ASRD_CHECKPOINT_FORMAT),
        )
    else:
        eval_e2e.main(argv, model_loader=model_from_checkpoint)


if __name__ == "__main__":
    main()
