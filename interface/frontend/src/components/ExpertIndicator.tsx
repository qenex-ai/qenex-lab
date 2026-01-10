import { Component } from 'solid-js'

interface ExpertIndicatorProps {
  name: string
  status: string
}

const ExpertIndicator: Component<ExpertIndicatorProps> = (props) => {
  return (
    <div class={`expert-indicator ${props.status}`}>
      {props.name}
    </div>
  )
}

export default ExpertIndicator
