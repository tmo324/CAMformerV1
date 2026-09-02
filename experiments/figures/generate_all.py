"""Regenerate all CAMformer paper figures included in this repository."""

from experiments.figures import generate_fig06, generate_fig08, generate_fig10


def main() -> int:
    for generate in (generate_fig06, generate_fig08, generate_fig10):
        print(f"Generated {generate()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
