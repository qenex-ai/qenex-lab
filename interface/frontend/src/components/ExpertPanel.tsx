import { Component, For } from 'solid-js'
import ExpertIndicator from './ExpertIndicator'

const EXPERTS = [
  "Physics", "Math", "Quantum", "Relativity", "Cosmology", "Thermo",
  "E&M", "Nuclear", "Particle", "Astro", "Materials", "Compute",
  "Info", "Stats", "Algebra", "Geometry", "Topology", "Analysis"
]

interface ExpertPanelProps {
  expertStatus: Record<string, string>
}

const ExpertPanel: Component<ExpertPanelProps> = (props) => {
  return (
    <div class="expert-panel">
      <h2>Trinity Expert System</h2>
      <div class="expert-count">18/18 Active</div>

      <div class="expert-grid">
        <For each={EXPERTS}>
          {(expert) => (
            <ExpertIndicator
              name={expert}
              status={props.expertStatus[expert] || "idle"}
            />
          )}
        </For>
      </div>

      <div class="expert-legend">
        <div class="legend-item">
          <span class="status-dot idle"></span>
          <span>Idle</span>
        </div>
        <div class="legend-item">
          <span class="status-dot thinking"></span>
          <span>Thinking</span>
        </div>
        <div class="legend-item">
          <span class="status-dot validated"></span>
          <span>Validated</span>
        </div>
      </div>
    </div>
  )
}

export default ExpertPanel
