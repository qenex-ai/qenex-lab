# QENEX LAB Chat Interface

**Version**: 3.0-INFINITY
**Status**: ✅ OPERATIONAL

Dedicated conversational web interface for QENEX LAB with real-time streaming, expert status indicators, and LaTeX rendering.

---

## Features

- **Conversational UI**: Web-based chat interface with streaming responses
- **Trinity Pipeline**: Automatic routing between DeepSeek (code) and Scout 17B (theory)
- **18-Expert Grid**: Real-time status indicators showing expert system activity
- **Markdown + LaTeX**: Beautiful rendering of equations and code
- **Command Integration**: `/publish <topic>` generates PDF publications
- **100% Local**: All processing on localhost, zero external dependencies

---

## Quick Start

### Launch the Chat Interface

```bash
qlab --chat
```

This will:
1. Start the Ollama proxy (if not running)
2. Launch the FastAPI backend on port 8765
3. Launch the Solid.js frontend on port 5173
4. Open the interface in your browser at http://localhost:5173

### Stop the Interface

Press `Ctrl+C` in the terminal to stop all services.

---

## Architecture

```
Frontend (Solid.js)           Backend (FastAPI)         Trinity Pipeline
http://localhost:5173   →   http://localhost:8765   →   Ollama (11435)
                                    ↓                        ↓
                              Expert System            Scout CLI
                              (WebSocket)              (Validator)
```

---

## Usage

### Basic Chat

Simply type your question or request:
- Code questions automatically route to DeepSeek
- Theory questions automatically route to Scout 17B
- Real-time streaming responses

### LaTeX Support

Inline equations: `$E = mc^2$`
Display equations: `$$\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}$$`

### Code Blocks

\```python
def calculate_energy(mass):
    c = 299792458  # Speed of light
    return mass * c**2
\```

### Publish Command

Generate a research paper PDF:
```
/publish Gravitational Wave Detection Methods
```

Output: `/opt/qenex_lab/publications/Gravitational_Wave_Detection_Methods_*.pdf`

---

## Expert Panel

The right sidebar shows all 18 experts with real-time status:

- **Idle** (blue): Expert ready
- **Thinking** (cyan, pulsing): Expert analyzing
- **Validated** (green): Validation complete

Expert domains:
- Physics, Math, Quantum, Relativity, Cosmology, Thermodynamics
- Electromagnetism, Nuclear, Particle, Astrophysics
- Materials, Computation, Information, Statistics
- Algebra, Geometry, Topology, Analysis

---

## Directory Structure

```
/opt/qenex_lab/interface/
├── backend/              # Python FastAPI server
│   ├── main.py           # Server entry point
│   ├── trinity_router.py # Automatic model routing
│   ├── expert_system.py  # 18-expert status tracking
│   ├── pdf_generator.py  # PDF generation
│   └── venv/             # Python virtual environment
├── frontend/             # Solid.js web application
│   ├── src/
│   │   ├── App.tsx       # Main application
│   │   ├── components/   # UI components
│   │   ├── hooks/        # Custom hooks
│   │   └── styles/       # QENEX LAB theme
│   └── dist/             # Production build (after `bun run build`)
└── README.md             # This file
```

---

## Development

### Backend Development

```bash
cd /opt/qenex_lab/interface/backend
source venv/bin/activate
uvicorn main:app --reload --port 8765
```

### Frontend Development

```bash
cd /opt/qenex_lab/interface/frontend
bun run dev
```

### Build for Production

```bash
cd /opt/qenex_lab/interface/frontend
bun run build
```

---

## API Endpoints

### Backend (http://localhost:8765)

- `GET /health` - Health check
- `POST /chat/message` - Streaming chat (SSE)
- `WS /ws` - WebSocket for expert status
- `POST /publish` - Generate PDF

### Frontend (http://localhost:5173)

- Web UI with Solid.js reactivity
- Connects to backend via HTTP + WebSocket

---

## Troubleshooting

### Backend not starting

Check logs:
```bash
tail -f /tmp/qenex-chat-backend.log
```

Restart:
```bash
qlab --chat
```

### Frontend not loading

Check logs:
```bash
tail -f /tmp/qenex-chat-frontend.log
```

Verify Vite dev server:
```bash
curl http://localhost:5173
```

### Ollama proxy issues

Restart proxy:
```bash
pkill -f ollama-proxy
python3 /tmp/ollama-proxy.py > /tmp/ollama-proxy.log 2>&1 &
```

### Expert status not updating

Check WebSocket connection in browser console (F12)

---

## Success Criteria

- [x] `qlab --chat` launches interface
- [x] Chat window displays streaming responses
- [x] Expert panel shows 18 experts with status
- [x] Automatic model routing (code/theory)
- [x] LaTeX equations render correctly
- [x] Code syntax highlighting
- [x] `/publish` command works
- [x] Dark mode QENEX LAB theme
- [x] 100% local operation

---

## Version History

### v3.0-INFINITY (2026-01-07)
- Initial release
- Trinity Pipeline integration
- Real-time expert status
- Markdown + LaTeX rendering
- Command-in-chat support

---

**QENEX LAB** - Sovereign Scientific Computing
100% Local | Air-Gapped | Zero External Dependencies
