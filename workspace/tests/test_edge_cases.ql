# Edge Case Test Suite for Q-Lang Interpreter

# 1. Division by Zero
print ">>> Test 1: Division by Zero"
x = 1.0 / 0.0

# 2. Dimensional Mismatch
print ">>> Test 2: Dimensional Mismatch"
y = 1.0 * m + 1.0 * s

# 3. Undefined Pipe Function
print ">>> Test 3: Undefined Pipe Function"
z = 10.0 |> nonexistent_func

# 4. Invalid Chemistry Simulation
print ">>> Test 4: Invalid Chemistry"
simulate chemistry InvalidAtom 0,0,0

# 5. Extreme Physics Simulation (Negative Temperature)
print ">>> Test 5: Extreme Physics"
simulate physics 10 100 -500
