import { Component, For, Show } from 'solid-js'

interface DiscoveryFile {
  name: string
  path: string
  relevance: number
}

interface Experts {
  lagrangian: boolean
  scout_cli: boolean
  qlang: boolean
}

interface ContextBadgeProps {
  discoveryFiles: DiscoveryFile[]
  experts: Experts
  processingTime?: number
}

const ContextBadge: Component<ContextBadgeProps> = (props) => {
  return (
    <div class="context-badge">
      {/* Discovery Files Section */}
      <Show when={props.discoveryFiles.length > 0}>
        <div class="badge-section">
          <span class="badge-icon" title="Discovery Files">📚</span>
          <span class="badge-label">Active Memory:</span>
          <div class="badge-items">
            <For each={props.discoveryFiles}>
              {(file) => (
                <span
                  class="badge-item discovery-file"
                  title={`${file.path}\nRelevance: ${file.relevance.toFixed(3)}`}
                >
                  {file.name.length > 30 ? file.name.substring(0, 30) + '...' : file.name}
                </span>
              )}
            </For>
          </div>
        </div>
      </Show>

      {/* Active Experts Section */}
      <div class="badge-section">
        <span class="badge-icon" title="Expert Systems">🧠</span>
        <span class="badge-label">Experts:</span>
        <div class="badge-items">
          <Show when={props.experts.lagrangian}>
            <span class="badge-item expert-badge" title="Unified Lagrangian field theory">
              Lagrangian
            </span>
          </Show>
          <Show when={props.experts.scout_cli}>
            <span class="badge-item expert-badge" title="18-expert physics validation">
              Scout CLI
            </span>
          </Show>
          <Show when={props.experts.qlang}>
            <span class="badge-item expert-badge" title="Q-Lang formal verification">
              Q-Lang
            </span>
          </Show>
          <Show when={!props.experts.scout_cli && !props.experts.qlang}>
            <span class="badge-item expert-badge-minimal" title="Fast mode - validation disabled">
              Fast Mode
            </span>
          </Show>
        </div>
      </div>

      {/* Processing Time */}
      <Show when={props.processingTime !== undefined}>
        <div class="badge-section">
          <span class="badge-icon" title="Context Processing Time">⚡</span>
          <span class="badge-label">Context gathered in:</span>
          <span class="processing-time">{props.processingTime.toFixed(2)}s</span>
        </div>
      </Show>

      {/* OMNI-AWARE Indicator */}
      <div class="omni-indicator">
        <span class="omni-icon" title="OMNI-AWARE Mode Active">🌐</span>
        <span class="omni-text">OMNI-AWARE v1.4.0-INFINITY</span>
      </div>
    </div>
  )
}

export default ContextBadge
