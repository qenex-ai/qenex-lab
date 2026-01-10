import { Component, createSignal, onMount } from 'solid-js'
import ChatWindow from './components/ChatWindow'
import ExpertPanel from './components/ExpertPanel'
import StatusBar from './components/StatusBar'
import { useWebSocket } from './hooks/useWebSocket'

const App: Component = () => {
  const [validationStatus, setValidationStatus] = createSignal(null)
  const { connect, expertStatus } = useWebSocket('ws://localhost:8765/ws')

  onMount(() => {
    connect()
  })

  return (
    <div class="qenex-lab-app">
      {/* Header */}
      <header class="qenex-header">
        <h1>QENEX LAB | Sovereign Intelligence</h1>
        <div class="version">v3.0-INFINITY</div>
      </header>

      {/* Main Layout */}
      <div class="main-layout">
        {/* Left: Chat Window */}
        <div class="chat-section">
          <ChatWindow
            onValidation={setValidationStatus}
          />
        </div>

        {/* Right: Expert Panel */}
        <div class="expert-section">
          <ExpertPanel
            expertStatus={expertStatus()}
          />
        </div>
      </div>

      {/* Bottom: Status Bar */}
      <StatusBar
        validation={validationStatus()}
      />
    </div>
  )
}

export default App
