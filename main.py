from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "prepare": ["data_loader.reference"],
    "physical": ["experiments.cop_benchmark"],
    "pressure": ["experiments.pp_validation", "experiments.pti_validation"],
    "mix": ["experiments.mix_validation"],
    "stepup": ["experiments.cop_benchmark_stepup"],
    "stepup-ceiling": ["experiments.stepup_pressure_ceiling"],
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        raise SystemExit(f"usage: python main.py {{{','.join(COMMANDS)}}} [args]")
    command, args = sys.argv[1], sys.argv[2:]
    for module in COMMANDS[command]:
        print(f"==> {module}", flush=True)
        sys.argv = [module, *args]
        importlib.import_module(module).main()


if __name__ == "__main__":
    main()
