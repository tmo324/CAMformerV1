#!/usr/bin/env python3
"""Compatibility wrapper for the installed CAMformer validation command."""

from camformer.cli.validate import main


if __name__ == "__main__":
    raise SystemExit(main())
