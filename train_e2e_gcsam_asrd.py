"""Train the opt-in ASRD model through the existing GCSAM loop."""

from models.shared_resnet_sr_asrd import build_model
from train_e2e_gcsam import main as train


def main():
    train(model_factory=build_model)


if __name__ == "__main__":
    main()
