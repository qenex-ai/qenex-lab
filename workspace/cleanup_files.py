
import os

# ==========================================
# Clean integrals.py
# ==========================================
int_path = 'packages/qenex-chem/src/integrals.py'
with open(int_path, 'r') as f:
    lines = f.readlines()

# Find the separator blocks
separator = "# ==========================================\n"
helpers = "# Derivative Helpers\n"

indices = []
for i, line in enumerate(lines):
    if line == separator and (i+1 < len(lines) and lines[i+1] == helpers):
        indices.append(i)

if len(indices) >= 2:
    # Keep everything before the first occurrence
    # Keep everything from the second occurrence onwards
    new_lines = lines[:indices[0]] + lines[indices[1]:]
    
    with open(int_path, 'w') as f:
        f.writelines(new_lines)
    print(f"Cleaned integrals.py: Removed lines {indices[0]} to {indices[1]}")
else:
    print("Warning: Could not find duplicate blocks in integrals.py")

# ==========================================
# Clean solver.py
# ==========================================
sol_path = 'packages/qenex-chem/src/solver.py'
with open(sol_path, 'r') as f:
    lines = f.readlines()

uhf_indices = []
for i, line in enumerate(lines):
    if line.strip().startswith("class UHFSolver(HartreeFockSolver):"):
        uhf_indices.append(i)

if len(uhf_indices) >= 2:
    # Keep up to first definition
    # Append from second definition
    final_lines = lines[:uhf_indices[0]] + lines[uhf_indices[-1]:]
    
    # Fix variable scope in UHFSolver
    # Look for "C_0 = X @ C_0_prime" in the newly constructed lines to inject init
    # We iterate through final_lines to find the injection point
    
    injection_point = -1
    for i, line in enumerate(final_lines):
        if "C_0 = X @ C_0_prime" in line and i > uhf_indices[0]: # Ensure we are in UHF part
            injection_point = i + 1
            indent = line[:line.find("C")] # Get indentation
            break
            
    if injection_point != -1:
        init_code = [
            f"{indent}eps_a = eps_0\n",
            f"{indent}eps_b = eps_0\n",
            f"{indent}current_E = 0.0\n"
        ]
        final_lines = final_lines[:injection_point] + init_code + final_lines[injection_point:]
        print("Injected variable initialization into UHFSolver")
    
    with open(sol_path, 'w') as f:
        f.writelines(final_lines)
    print(f"Cleaned solver.py: Removed first UHFSolver definition")
else:
    print("Warning: Could not find duplicate UHFSolver in solver.py")
