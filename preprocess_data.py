"""CLI wrapper for preprocessing entry point."""

from scalovit.data import preprocess, get_args


if __name__ == "__main__":
    args = get_args()
    preprocess(args)
