import { Component, createSignal, For } from 'solid-js'
import MessageBubble from './MessageBubble'
import PromptInput from './PromptInput'
import ContextBadge from './ContextBadge'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: number
  context?: any  // OMNI-AWARE context metadata
}

interface ChatWindowProps {
  onValidation: (v: any) => void
}

const ChatWindow: Component<ChatWindowProps> = (props) => {
  const [messages, setMessages] = createSignal<Message[]>([])
  const [isStreaming, setIsStreaming] = createSignal(false)

  const sendMessage = async (content: string) => {
    // Add user message
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content,
      timestamp: Date.now()
    }
    setMessages([...messages(), userMsg])

    // Start streaming assistant response
    setIsStreaming(true)
    let assistantContent = ''
    let contextMetadata: any = null

    const assistantMsg: Message = {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: '',
      timestamp: Date.now()
    }
    setMessages([...messages(), assistantMsg])

    try {
      // Use /chat/simple endpoint: User → DeepSeek → Scout 17B (automatic)
      const response = await fetch('http://localhost:8765/chat/simple', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content, enable_validation: false })
      })

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      while (reader) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            try {
              const parsed = JSON.parse(data)
              if (parsed.content) {
                assistantContent += parsed.content
                // OPTIMIZED: Indexed update instead of array cloning
                setMessages(msgs => {
                  const lastIdx = msgs.length - 1
                  const updated = [...msgs]
                  updated[lastIdx] = { ...updated[lastIdx], content: assistantContent, context: contextMetadata }
                  return updated
                })
              }
            } catch {}
          } else if (line.startsWith('event: context')) {
            // Handle OMNI-AWARE context event
            const nextLine = lines[lines.indexOf(line) + 1]
            if (nextLine?.startsWith('data: ')) {
              try {
                contextMetadata = JSON.parse(nextLine.slice(6))
                console.log('[ChatWindow] Received OMNI-AWARE context:', contextMetadata)
              } catch {}
            }
          } else if (line.startsWith('event: validation')) {
            // Handle validation event
            const nextLine = lines[lines.indexOf(line) + 1]
            if (nextLine?.startsWith('data: ')) {
              try {
                const validation = JSON.parse(nextLine.slice(6))
                props.onValidation(validation)
              } catch {}
            }
          } else if (line.startsWith('event: tool_call')) {
            // DeepSeek is calling Scout 17B
            const nextLine = lines[lines.indexOf(line) + 1]
            if (nextLine?.startsWith('data: ')) {
              try {
                const toolCall = JSON.parse(nextLine.slice(6))
                console.log('[ChatWindow] Tool call:', toolCall.tool)
                // Could show "Consulting Scout 17B..." indicator
              } catch {}
            }
          } else if (line.startsWith('event: tool_result')) {
            // Scout 17B responded
            const nextLine = lines[lines.indexOf(line) + 1]
            if (nextLine?.startsWith('data: ')) {
              try {
                const toolResult = JSON.parse(nextLine.slice(6))
                console.log('[ChatWindow] Tool result:', toolResult.result)
                // Could show "Scout consulted, synthesizing..." indicator
              } catch {}
            }
          }
        }
      }
    } catch (error) {
      console.error('Streaming error:', error)
    }

    setIsStreaming(false)
  }

  return (
    <div class="chat-window">
      <div class="messages-container">
        <For each={messages()}>
          {(msg) => (
            <>
              {/* Show ContextBadge for assistant messages with OMNI-AWARE context */}
              {msg.role === 'assistant' && msg.context && (
                <ContextBadge
                  discoveryFiles={msg.context.discovery_files || []}
                  experts={msg.context.experts || {}}
                  processingTime={msg.context.processing_time}
                />
              )}
              <MessageBubble message={msg} />
            </>
          )}
        </For>
      </div>

      <PromptInput
        onSend={sendMessage}
        disabled={isStreaming()}
      />
    </div>
  )
}

export default ChatWindow
