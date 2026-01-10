import { Component, createMemo, createSignal, onCleanup } from 'solid-js'
import { marked } from 'marked'
import markedKatex from 'marked-katex-extension'

// Configure marked with KaTeX
marked.use(markedKatex({
  throwOnError: false,
  output: 'html'
}))

interface MarkdownRendererProps {
  content: string
}

const MarkdownRenderer: Component<MarkdownRendererProps> = (props) => {
  const [debouncedContent, setDebouncedContent] = createSignal(props.content)
  let timeoutId: number | undefined

  // Debounce: Only re-parse after 100ms of no updates
  createMemo(() => {
    const content = props.content
    if (timeoutId) clearTimeout(timeoutId)

    timeoutId = setTimeout(() => {
      setDebouncedContent(content)
    }, 100) as unknown as number
  })

  onCleanup(() => {
    if (timeoutId) clearTimeout(timeoutId)
  })

  const html = createMemo(() => {
    try {
      return marked.parse(debouncedContent())
    } catch (error) {
      console.error('Markdown parsing error:', error)
      return `<pre>${debouncedContent()}</pre>`
    }
  })

  return (
    <div
      class="markdown-content"
      innerHTML={html()}
    />
  )
}

export default MarkdownRenderer
