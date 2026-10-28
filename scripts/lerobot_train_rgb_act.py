#!/usr/bin/env python3
"""Run LeRobot ACT with a saved image-resize preprocessing step."""
import argparse
import sys


def add_image_resize(preprocessor, image_size):
    from lerobot.processor import ImageCropResizeProcessorStep, NormalizerProcessorStep

    for step in preprocessor.steps:
        if isinstance(step, ImageCropResizeProcessorStep):
            if tuple(step.resize_size or ()) != (image_size, image_size):
                raise ValueError("Checkpoint image size differs from the requested image size")
            return preprocessor
    # Resize after device placement, before normalization, using LeRobot's serializable step.
    index = next(i for i, step in enumerate(preprocessor.steps) if isinstance(step, NormalizerProcessorStep))
    preprocessor.steps = [*preprocessor.steps[:index],
                          ImageCropResizeProcessorStep(resize_size=(image_size, image_size)),
                          *preprocessor.steps[index:]]
    return preprocessor


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--image-size", type=int, required=True)
    args, trainer_args = parser.parse_known_args()
    if args.image_size <= 0:
        parser.error("image size must be positive")
    sys.argv = [sys.argv[0], *trainer_args]

    from lerobot.scripts import lerobot_train

    original = lerobot_train.make_pre_post_processors

    def make_processors(*a, **kw):
        preprocessor, postprocessor = original(*a, **kw)
        return add_image_resize(preprocessor, args.image_size), postprocessor

    lerobot_train.make_pre_post_processors = make_processors
    print(f"RGB preprocessing: resize to {args.image_size}x{args.image_size} before normalization", flush=True)
    lerobot_train.main()


if __name__ == "__main__":
    main()
