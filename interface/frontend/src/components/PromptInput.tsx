import { Component, createSignal } from 'solid-js'

interface PromptInputProps {
  onSend: (message: string) => Promise<void>
  disabled: boolean
}

const PromptInput: Component<PromptInputProps> = (props) => {
  const [input, setInput] = createSignal('')

  const handleSubmit = async (e: Event) => {
    e.preventDefault()
    const text = input().trim()
    if (!text || props.disabled) return

    // Check for /publish command
    if (text.startsWith('/publish')) {
      const topic = text.replace('/publish', '').trim()
      await fetch('http://localhost:8765/publish', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ topic })
      })
      setInput('')
      return
    }

    await props.onSend(text)
    setInput('')
  }

  return (
    <form class="prompt-input" onSubmit={handleSubmit}>
      <input
        type="text"
        value={input()}
        onInput={(e) => setInput(e.currentTarget.value)}
        placeholder="Ask anything... (or /publish <topic>)"
        disabled={props.disabled}
      />
      <button type="submit" disabled={props.disabled || !input().trim()}>
        Send
      </button>
    </form>
  )
}

export default PromptInput
