# 🚀 QENEX LAB OMNI_INTEGRATION - STARTUP GUIDE

## Quick Start (Two Commands)

### Terminal 1 - Backend (OMNI-AWARE Intelligence)
```bash
cd /opt/qenex_lab/interface/backend
./start_omni.sh
```

### Terminal 2 - Frontend (UI)
```bash
cd /opt/qenex_lab/interface/frontend
./start_frontend.sh
```

Then visit: **http://localhost:5173** or **http://51.89.192.20:5173**

---

## Alternative: Manual Start

### Backend (Terminal 1)
```bash
cd /opt/qenex_lab/interface/backend
source venv/bin/activate
python main.py
```

### Frontend (Terminal 2)
```bash
cd /opt/qenex_lab/interface/frontend
/usr/bin/bun run dev
```

---

## What Happens on Startup

### Backend Initialization (~15 seconds)
```
[Context Bridge] Initializing Global Context Bridge...
[Context Bridge] Indexed: quantum_gravity_unification_v1.0.json
[Context Bridge] Indexed: SYSTEM_MANIFEST.json
...
[Context Bridge] ✓ Indexed 81 documents
[Multi-Expert] ✓ Scout CLI available
[Multi-Expert] ✓ Q-Lang available
OMNI_INTEGRATION INITIALIZATION COMPLETE
System is now OMNI-AWARE

INFO:     Uvicorn running on http://0.0.0.0:8765
```

### Frontend Startup (~1 second)
```
VITE v6.4.1  ready in 447 ms
➜  Local:   http://localhost:5173/
➜  Network: http://51.89.192.20:5173/
```

---

## Verify It's Working

### 1. Check Backend Health
```bash
curl http://localhost:8765/health | python -m json.tool
```

Expected response:
```json
{
  "status": "operational",
  "version": "1.4.0-INFINITY (OMNI-AWARE)",
  "lab": "QENEX LAB",
  "omni_integration": {
    "context_bridge": {
      "total_documents": 81,
      "faiss_vectors": 81
    },
    "multi_expert": {
      "scout_cli_available": true,
      "qlang_available": true
    }
  }
}
```

### 2. Open Frontend
Visit: http://localhost:5173

You should see:
- QENEX LAB header
- Chat interface
- Expert status panel (18 experts)

---

## Test OMNI-AWARE Features

Send these queries to see the system in action:

### Query 1: Quantum Gravity
```
Explain quantum gravity unification
```

**Expected Context Badge:**
```
📚 Active Memory: quantum_gravity_unification_v1.0.json, QENEX_SCOUT...
🧠 Experts: Lagrangian · Fast Mode
⚡ Context gathered in: 0.21s
🌐 OMNI-AWARE v1.4.0-INFINITY
```

### Query 2: Superconductivity
```
How does LaH10 achieve superconductivity at high pressure?
```

**Expected Context Badge:**
```
📚 Active Memory: RESEARCH_PAPER_LaH10_Superconductivity.md, lah10.json
🧠 Experts: Lagrangian · Fast Mode
```

### Query 3: Trinity Architecture
```
What is the Trinity Pipeline?
```

**Expected Context Badge:**
```
📚 Active Memory: trinity_orchestrator_v1.0.json, /opt/qenex/brain/README.md
🧠 Experts: Lagrangian · Fast Mode
```

---

## Troubleshooting

### Backend won't start
```bash
# Check permissions
ls -la /opt/qenex_lab/interface/backend/main.py

# Should show: -rw-r--r-- ubuntu ubuntu

# Fix if needed:
sudo chown -R ubuntu:ubuntu /opt/qenex_lab/interface/backend
```

### Frontend won't start
```bash
# Check if bun is available
/usr/bin/bun --version
# Should show: 1.3.5

# Try direct command:
cd /opt/qenex_lab/interface/frontend
/usr/bin/bun run dev
```

### Port already in use
```bash
# Backend (port 8765)
sudo lsof -i :8765
sudo kill -9 <PID>

# Frontend (port 5173)
sudo lsof -i :5173
sudo kill -9 <PID>
```

---

## Understanding the UI

### ContextBadge (appears above assistant responses)

**Blue badges** = Discovery files
- Hover to see: full path + relevance score
- Example: `quantum_gravity_unification_v1.0.json (0.580)`

**Green badges** = Expert systems
- `Lagrangian` = Unified Lagrangian loaded
- `Scout CLI` = 18-expert physics validation (when enabled)
- `Q-Lang` = Formal verification (when enabled)
- `Fast Mode` = Default (no validation, faster)

**Processing time** = How long context gathering took
- Typical: 100-200ms

**OMNI-AWARE badge** = System is using full knowledge base

---

## Performance Notes

### First Query (Cold Start)
- Backend indexing: ~15 seconds (one-time)
- Model loading: ~3-5 seconds
- Total: ~20 seconds

### Subsequent Queries
- Cache hit: <100ms (instant)
- Cache miss: 2-3 seconds (context + inference)
- With validation: 4-5 seconds (+Scout CLI)

### Memory Usage
- Backend: ~2GB (sentence-transformers + FAISS)
- Frontend: ~100MB (Vite dev server)
- Models (Ollama): ~30GB (DeepSeek + Scout 17B)

---

## Stopping the System

### Graceful Shutdown
Press `Ctrl+C` in each terminal

### Force Kill
```bash
# Backend
pkill -f "python main.py"

# Frontend
pkill -f "vite --port 5173"
```

---

## Next Steps

1. **Try example queries** (see above)
2. **Watch ContextBadge** update with relevant discoveries
3. **Enable validation mode** for rigorous physics checking:
   - Edit `ChatWindow.tsx` line 49
   - Change `enable_validation: false` to `enable_validation: true`
   - Rebuild frontend: `bun run build`

4. **Explore indexed knowledge**:
   ```bash
   cd /opt/qenex_lab/interface/backend
   source venv/bin/activate
   python test_omni_startup.py
   ```

---

## Documentation

- **Full implementation**: `/tmp/OMNI_INTEGRATION_COMPLETE.md`
- **Plan**: `/root/.claude/plans/typed-soaring-acorn.md`
- **API docs**: See health endpoint response

---

**Status**: ✅ OPERATIONAL
**Version**: v1.4.0-INFINITY
**Mode**: OMNI-AWARE

The system is ready to demonstrate maximum Intelligence Threshold!
