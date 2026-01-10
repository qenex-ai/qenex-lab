import { createSignal, onCleanup } from 'solid-js'

export function useWebSocket(url: string) {
  const [ws, setWs] = createSignal<WebSocket | null>(null)
  const [expertStatus, setExpertStatus] = createSignal<Record<string, string>>({})

  const connect = () => {
    const socket = new WebSocket(url)

    socket.onopen = () => {
      console.log('[WebSocket] Connected')
      setWs(socket)
    }

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'expert_status') {
        setExpertStatus(data.experts)
      }
    }

    socket.onerror = (error) => {
      console.error('[WebSocket] Error:', error)
    }

    socket.onclose = () => {
      console.log('[WebSocket] Disconnected')
      setWs(null)
    }

    onCleanup(() => {
      socket.close()
    })
  }

  return { connect, expertStatus }
}
