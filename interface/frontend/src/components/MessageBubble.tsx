import { Component } from 'solid-js'
import MarkdownRenderer from './MarkdownRenderer'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: number
}

interface MessageBubbleProps {
  message: Message
}

const MessageBubble: Component<MessageBubbleProps> = (props) => {
  return (
    <div class={`message-bubble ${props.message.role}`}>
      <MarkdownRenderer content={props.message.content} />
      {props.message.role === 'assistant' && (
        <div class="trinity-signature"></div>
      )}
    </div>
  )
}

export default MessageBubble
