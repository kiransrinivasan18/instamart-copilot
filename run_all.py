"""Optional convenience runner: executes every step in order, stopping at the first failure.
The README's numbered Reproduce commands remain the primary way to run the project."""
import subprocess
import sys

STEPS = ["generate_data.py", "part_a.py", "validate.py", "run_sql.py",
         "part_b.py", "part_c.py", "part_d.py"]

for step in STEPS:
    print(f"\n>>> {step}", flush=True)
    if subprocess.run([sys.executable, step]).returncode != 0:
        sys.exit(f"{step} failed - stopping.")
print("\nAll steps finished.")
