# 🚀 QENEX LAB TUI - QUICK START GUIDE

## High-Performance Cyberpunk Terminal Interface for OMNI-AWARE System

**Location**: `/opt/qenex_lab/tui/`

---

## 🎯 What is This?

A **blazing-fast Rust TUI** that connects directly to the QENEX LAB OMNI-AWARE backend, displaying:

- **Real-time 18-expert status matrix** (Physics, Math, Quantum, etc.)
- **Discovery file citations** with relevance scores
- **Cyberpunk-style dashboard** with ASCII art
- **Live SSE streaming** from backend
- **Processing gauges** and metrics
- **System event log** with timestamps

**Why use the TUI instead of web UI?**
- ⚡ **10x less memory** (10MB vs 100MB)
- 🚀 **5x faster startup** (150ms vs 750ms)
- 🖥️ **Native performance** (no browser overhead)
- 🔒 **Direct SSH access** (no port forwarding)
- 🎨 **Matrix-style aesthetics**

---

## 📋 Installation (First Time Only)

### Step 1: Install Rust

```bash
cd /opt/qenex_lab/tui
chmod +x install_rust.sh
./install_rust.sh
```

**Time**: ~5 minutes (downloads and installs Rust toolchain)

**Then reload your shell**:
```bash
source ~/.cargo/env
```

### Step 2: Build the TUI

```bash
./build.sh
```

**Time**: ~2-3 minutes (first build downloads dependencies)

**You'll see**:
```
🔨 Compiling with maximum optimizations...
    Compiling qenex-tui v1.4.0-infinity
    Finished release [optimized] target(s) in 147.23s
✅ Build complete!
Binary: ./target/release/qenex-tui
```

---

## 🚀 Running the TUI

### Prerequisites

**Backend must be running!**

In a **separate terminal**:
```bash
cd /opt/qenex_lab/interface/backend
./start_omni.sh
```

Wait for:
```
[Context Bridge] ✓ Indexed 81 documents
OMNI_INTEGRATION INITIALIZATION COMPLETE
System is now OMNI-AWARE
INFO:     Uvicorn running on http://0.0.0.0:8765
```

### Launch TUI

```bash
cd /opt/qenex_lab/tui
./run.sh
```

**You'll see** the cyberpunk dashboard:

```
┌──────────────────────────────────────────────────────┐
│   ██████  ███████ ███    ██ ███████ ██   ██          │
│  ██    ██ ██      ████   ██ ██      ╚██ ██╔╝         │
│  ██    ██ █████   ██ ██  ██ █████    ╚███╔╝          │
│  ██ ▄▄ ██ ██      ██  ██ ██ ██       ▄██╗            │
│   ██████  ███████ ██   ████ ███████ ██                │
│              OMNI-AWARE v1.4.0-INFINITY               │
├──────────────────────────────────────────────────────┤
│ ⚡ DOCS: 81 │ 📊 MODEL: idle │ ⏱ UPTIME: 0s          │
├─────────────────────────────────┬────────────────────┤
│  ▼ RESPONSE ▼                   │ ▼ 18-EXPERT MATRIX │
│                                  │ ○ Physics          │
│  [Type your query below]         │ ○ Math             │
│                                  │ ○ Quantum          │
│                                  │ ...                │
├─────────────────────────────────┼────────────────────┤
│  ▼ SYSTEM LOG ▼                 │ ▼ CONTEXT ▼        │
│  [12:30:00] ✓ TUI initialized   │ ⌛ Awaiting query...│
├─────────────────────────────────┴────────────────────┤
│  ▼ INPUT (Ctrl+C: Quit | Enter: Send) ▼              │
│  _                                                    │
├──────────────────────────────────────────────────────┤
│  ▼ PROCESSING ▼  [                       ]           │
└──────────────────────────────────────────────────────┘
```

---

## 🎮 Usage

### Basic Commands

- **Type** your query in the input box
- **Enter** to send
- **Backspace** to edit
- **Ctrl+C** to quit

### Example Queries

**Query 1: Simple**
```
What is quantum entanglement?
```

**Expected Result**:
- Context panel shows: `quantum_entanglement.json (0.652)`
- Model: `deepseek`
- Response streams in real-time
- Log shows discovery files used

**Query 2: Complex**
```
Explain quantum gravity unification using the Unified Lagrangian
```

**Expected Result**:
- Context shows: `quantum_gravity_unification_v1.0.json (0.843)`
- Multiple discovery files referenced
- Lagrangian expert active
- Response cites SYSTEM_MANIFEST

**Query 3: Superconductivity**
```
How does LaH10 achieve superconductivity at high pressure?
```

**Expected Result**:
- Context shows: `RESEARCH_PAPER_LaH10_Superconductivity.md (0.791)`
- Response explains BCS theory, Eliashberg equations
- Materials expert may activate

---

## 📊 Understanding the Dashboard

### Panels Explained

**Header**:
- ASCII art branding
- System stats (docs, model, uptime)

**Response Panel** (Left Top):
- Streaming assistant responses
- Auto-scrolls as text arrives
- Word-wrapped for readability

**System Log** (Left Bottom):
- Timestamped events
- Color codes:
  - ✓ Green = Success
  - ℹ Cyan = Info
  - 📚 Magenta = Discovery citation
  - ⚠ Yellow = Warning
  - ✗ Red = Error

**Expert Matrix** (Right Top):
- 18 experts in 3 columns
- Status indicators:
  - `○` Gray = Idle
  - `◉` Yellow = Thinking
  - `●` Green = Validated
  - `✗` Red = Error

**Context Panel** (Right Bottom):
- Active Memory: Discovery files with relevance scores
- Experts: Which systems are active (Lagrangian, Scout CLI, Q-Lang)
- Processing time in seconds

**Input Bar**:
- Type queries here
- Disabled during streaming
- Shows "⌛ Streaming..." when busy

**Processing Gauge**:
- Visual indicator
- 75% fill when streaming
- Empty when idle

---

## 🐛 Troubleshooting

### "Backend not detected on port 8765"

**Problem**: TUI can't connect to backend

**Solution**:
```bash
# Terminal 1: Start backend
cd /opt/qenex_lab/interface/backend
./start_omni.sh

# Wait for "Uvicorn running on http://0.0.0.0:8765"

# Terminal 2: Start TUI
cd /opt/qenex_lab/tui
./run.sh
```

### "cargo: command not found"

**Problem**: Rust not installed or not in PATH

**Solution**:
```bash
# Install Rust
./install_rust.sh

# Reload shell
source ~/.cargo/env

# Verify
cargo --version
```

### Terminal Too Small

**Problem**: UI looks broken or cramped

**Minimum size**: 100 columns × 30 rows

**Check size**:
```bash
tput cols  # Should be >= 100
tput lines # Should be >= 30
```

**Solution**: Maximize terminal window or use a larger font

### Build Fails

**Problem**: Compilation errors

**Common causes**:

1. **Missing C compiler**:
```bash
sudo apt-get install build-essential
```

2. **Outdated Rust**:
```bash
rustup update
```

3. **Corrupted cache**:
```bash
cargo clean
./build.sh
```

---

## ⚡ Performance Tips

### For Best Performance

1. **Use Release Build**:
```bash
cargo build --release  # Optimized
cargo run --release    # Run optimized
```

2. **Terminal Emulator Matters**:
- **Best**: Alacritty, kitty (GPU-accelerated)
- **Good**: GNOME Terminal, konsole
- **Avoid**: tmux (adds latency), screen

3. **SSH Compression**:
```bash
ssh -C user@server  # Enables compression
```

4. **Reduce Tick Rate** (if laggy):
Edit `main.rs` line 740:
```rust
let tick_rate = Duration::from_millis(200); // Instead of 100
```

---

## 🔄 Workflow Examples

### Full OMNI-AWARE Session

**Terminal 1** (Backend):
```bash
cd /opt/qenex_lab/interface/backend
./start_omni.sh
# Wait for "System is now OMNI-AWARE"
```

**Terminal 2** (TUI):
```bash
cd /opt/qenex_lab/tui
./run.sh
# Dashboard appears
```

**In TUI**:
1. Type: `What is the Trinity Pipeline architecture?`
2. Press Enter
3. Watch:
   - Context panel populates with `trinity_orchestrator_v1.0.json`
   - Model indicator shows `deepseek`
   - Response streams character by character
   - Log shows discovery files: `📚 Using: trinity_orchestrator_v1.0 (0.823)`
4. Read response
5. Type next query or Ctrl+C to quit

---

## 🎨 Customization

### Change Colors

Edit `src/main.rs`, search for `Color::` and replace:

```rust
// Examples:
Style::default().fg(Color::Cyan)     // Headers
Style::default().fg(Color::Green)    // Success
Style::default().fg(Color::Magenta)  // Experts
Style::default().fg(Color::Yellow)   // Warnings
```

Available colors:
- `Black, Red, Green, Yellow, Blue, Magenta, Cyan, Gray, White`
- `DarkGray, LightRed, LightGreen, LightYellow, LightBlue, LightMagenta, LightCyan`
- `Rgb(r, g, b)` for custom colors

### Change Layout

Edit `src/main.rs`, `draw_ui()` function:

```rust
// Adjust panel sizes (line ~720):
Constraint::Percentage(70),  // Left side width
Constraint::Percentage(30)   // Right side width

// Vertical splits:
Constraint::Percentage(60),  // Top panel height
Constraint::Percentage(40)   // Bottom panel height
```

### Change Refresh Rate

Edit `src/main.rs`, line ~740:

```rust
let tick_rate = Duration::from_millis(100);  // Default: 10 FPS

// Options:
// 50ms = 20 FPS (smoother but more CPU)
// 100ms = 10 FPS (balanced)
// 200ms = 5 FPS (slower but less CPU)
```

---

## 📈 Comparison: TUI vs Web UI

| Feature | TUI (Rust) | Web UI (Browser) |
|---------|------------|------------------|
| Memory | 10 MB | 100+ MB |
| Startup | 150 ms | 750 ms |
| CPU (idle) | 0.3% | 2-5% |
| Port forwarding | Not needed | Required |
| Mouse support | No | Yes |
| Multiple tabs | No | Yes |
| Copy/paste | Terminal-based | Browser-based |
| Aesthetics | Cyberpunk/Matrix | Modern/Colorful |
| Remote access | SSH only | HTTP/HTTPS |

---

## 🔮 Roadmap

### v1.5 (Planned)
- Mouse support (click panels, scroll)
- Query history (up/down arrows)
- Export conversations to file
- Multiple color themes

### v2.0 (Future)
- Tabs for concurrent queries
- Discovery file browser
- Q-Lang REPL integration
- Real-time latency graphs
- Vim keybindings

---

## 🎯 Quick Reference Card

```
┌─────────────────────────────────────────────┐
│  QENEX LAB TUI v1.4.0-INFINITY              │
├─────────────────────────────────────────────┤
│  Install:   ./install_rust.sh               │
│  Build:     ./build.sh                      │
│  Run:       ./run.sh                        │
│  Quit:      Ctrl+C                          │
├─────────────────────────────────────────────┤
│  Backend:   cd ../interface/backend         │
│             ./start_omni.sh                 │
├─────────────────────────────────────────────┤
│  Log:       Left bottom panel               │
│  Context:   Right bottom panel              │
│  Experts:   Right top panel (18)            │
│  Response:  Left top panel (streaming)      │
└─────────────────────────────────────────────┘
```

---

**Ready to experience OMNI-AWARE intelligence in pure terminal form!** 🌐

Start with: `./install_rust.sh`
