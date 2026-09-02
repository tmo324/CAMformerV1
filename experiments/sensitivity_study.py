#!/usr/bin/env python3
"""Compatibility wrapper for the installed CAMformer sweep command."""

from camformer.cli.sweep import main


if __name__ == "__main__":
    raise SystemExit(main())
