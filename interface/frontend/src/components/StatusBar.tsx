import { Component, Show } from 'solid-js'

interface StatusBarProps {
  validation: any
}

const StatusBar: Component<StatusBarProps> = (props) => {
  return (
    <div class="status-bar">
      <Show when={props.validation} fallback={<span>Lagrangian Gate: Standby</span>}>
        <span>
          Lagrangian Gate: {props.validation.valid ? '✓ VALIDATED' : '✗ INVALID'}
          {' '}
          (Confidence: {(props.validation.confidence * 100).toFixed(1)}%)
        </span>
      </Show>
    </div>
  )
}

export default StatusBar
