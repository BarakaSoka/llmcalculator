"""Entry point for the standalone build.

A user who double-clicks the packaged binary gets the web app, because that is
the interface that needs no prior knowledge. Anyone passing arguments clearly
wants the CLI, so give them that instead.
"""

import sys


def main() -> int:
    from llmcalculator.cli import main as cli_main

    if len(sys.argv) > 1:
        return cli_main()
    return cli_main(["app"])


if __name__ == "__main__":
    sys.exit(main())
