use anyhow::Result;
use chrono::Local;
use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers,
        DisableBracketedPaste, EnableBracketedPaste,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{
        Block, Borders, Gauge, List, ListItem, Paragraph, Wrap,
    },
    Frame, Terminal,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{
    io,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, Mutex};

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExpertStatus {
    #[serde(flatten)]
    experts: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone)]
struct DiscoveryFile {
    name: String,
    path: String,
    relevance: f64,
}

#[derive(Debug, Clone)]
struct ContextMetadata {
    discovery_files: Vec<DiscoveryFile>,
    experts: ExpertData,
    processing_time: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExpertData {
    lagrangian: bool,
    scout_cli: bool,
    qlang: bool,
}

#[derive(Debug, Clone)]
enum UIEvent {
    Message(String),
    Context(ContextMetadata),
    Model(String),
    ExpertStatus(ExpertStatus),
    ToolCall(String),  // Tool being called (e.g., "scout")
    ToolResult(String), // Tool result summary
    Done,
    Error(String),
}

#[derive(Debug, Clone)]
struct LogEntry {
    timestamp: String,
    message: String,
    level: LogLevel,
}

#[derive(Debug, Clone)]
enum LogLevel {
    Info,
    Success,
    Warning,
    Error,
    Discovery,
}

// ============================================================================
// APPLICATION STATE
// ============================================================================

struct AppState {
    messages: Vec<String>,
    input: String,
    log: Vec<LogEntry>,
    expert_status: ExpertStatus,
    current_context: Option<ContextMetadata>,
    current_model: Option<String>,
    is_streaming: bool,
    processing_time: f64,
    total_documents: usize,
    last_response_time: Duration,
    start_time: Instant,
    response_scroll: usize,  // Scroll offset for response view
}

impl AppState {
    fn new() -> Self {
        let mut experts = std::collections::HashMap::new();
        for expert in [
            "Physics", "Math", "Quantum", "Relativity", "Cosmology", "Thermo",
            "E&M", "Nuclear", "Particle", "Astro", "Materials", "Compute",
            "Info", "Stats", "Algebra", "Geometry", "Topology", "Analysis",
        ] {
            experts.insert(expert.to_string(), "idle".to_string());
        }

        Self {
            messages: vec![],
            input: String::new(),
            log: vec![LogEntry {
                timestamp: Local::now().format("%H:%M:%S").to_string(),
                message: "QENEX LAB OMNI-AWARE TUI v1.4.0-INFINITY initialized".to_string(),
                level: LogLevel::Success,
            }],
            expert_status: ExpertStatus { experts },
            current_context: None,
            current_model: None,
            is_streaming: false,
            processing_time: 0.0,
            total_documents: 81,
            last_response_time: Duration::from_secs(0),
            start_time: Instant::now(),
            response_scroll: 0,
        }
    }

    fn add_log(&mut self, message: String, level: LogLevel) {
        self.log.push(LogEntry {
            timestamp: Local::now().format("%H:%M:%S").to_string(),
            message,
            level,
        });

        // Keep only last 100 entries
        if self.log.len() > 100 {
            self.log.drain(0..20);
        }
    }
}

// ============================================================================
// BACKEND COMMUNICATION
// ============================================================================

async fn send_query(
    query: String,
    tx: mpsc::Sender<UIEvent>,
) -> Result<()> {
    let client = Client::new();

    // Send POST request to /chat/simple (User → DeepSeek → Scout)
    let response = client
        .post("http://localhost:8765/chat/simple")
        .json(&serde_json::json!({
            "content": query,
            "enable_validation": false
        }))
        .send()
        .await?;

    // Stream SSE events
    let mut stream = response.bytes_stream();
    use futures::StreamExt;

    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let text = String::from_utf8_lossy(&chunk);
        buffer.push_str(&text);

        // Process complete lines
        while let Some(pos) = buffer.find('\n') {
            let line = buffer.drain(..=pos).collect::<String>();
            let line = line.trim();

            if line.starts_with("event: ") {
                let event_type = line.strip_prefix("event: ").unwrap_or("");

                // For "done" event, we can process immediately without waiting for data
                if event_type == "done" {
                    let _ = tx.send(UIEvent::Done).await;
                    continue;
                }

                // For other events, wait for the data line
                if let Some(data_pos) = buffer.find('\n') {
                    let data_line = buffer.drain(..=data_pos).collect::<String>();
                    if let Some(data) = data_line.trim().strip_prefix("data: ") {
                        match event_type {
                            "context" => {
                                if let Ok(context_data) = serde_json::from_str::<serde_json::Value>(data) {
                                    let discovery_files = context_data["discovery_files"]
                                        .as_array()
                                        .map(|arr| {
                                            arr.iter()
                                                .filter_map(|f| {
                                                    Some(DiscoveryFile {
                                                        name: f["name"].as_str()?.to_string(),
                                                        path: f["path"].as_str()?.to_string(),
                                                        relevance: f["relevance"].as_f64()?,
                                                    })
                                                })
                                                .collect()
                                        })
                                        .unwrap_or_default();

                                    let experts = serde_json::from_value(context_data["experts"].clone())
                                        .unwrap_or(ExpertData {
                                            lagrangian: false,
                                            scout_cli: false,
                                            qlang: false,
                                        });

                                    let processing_time = context_data["processing_time"].as_f64();

                                    tx.send(UIEvent::Context(ContextMetadata {
                                        discovery_files,
                                        experts,
                                        processing_time,
                                    }))
                                    .await?;
                                }
                            }
                            "model" => {
                                if let Ok(model_data) = serde_json::from_str::<serde_json::Value>(data) {
                                    if let Some(model) = model_data["model"].as_str() {
                                        tx.send(UIEvent::Model(model.to_string())).await?;
                                    }
                                }
                            }
                            "message" => {
                                if let Ok(msg_data) = serde_json::from_str::<serde_json::Value>(data) {
                                    if let Some(content) = msg_data["content"].as_str() {
                                        tx.send(UIEvent::Message(content.to_string())).await?;
                                    }
                                }
                            }
                            "error" => {
                                if let Ok(err_data) = serde_json::from_str::<serde_json::Value>(data) {
                                    if let Some(error) = err_data["error"].as_str() {
                                        tx.send(UIEvent::Error(error.to_string())).await?;
                                    }
                                }
                            }
                            "tool_call" => {
                                if let Ok(tool_data) = serde_json::from_str::<serde_json::Value>(data) {
                                    if let Some(tool) = tool_data["tool"].as_str() {
                                        tx.send(UIEvent::ToolCall(tool.to_string())).await?;
                                    }
                                }
                            }
                            "tool_result" => {
                                if let Ok(result_data) = serde_json::from_str::<serde_json::Value>(data) {
                                    if let Some(result) = result_data["result"].as_str() {
                                        tx.send(UIEvent::ToolResult(result.to_string())).await?;
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

// ============================================================================
// UI RENDERING
// ============================================================================

fn render_ascii_art() -> Vec<Line<'static>> {
    vec![
        Line::from(vec![
            Span::styled("   ██████  ███████ ███    ██ ███████ ██   ██     ██       █████  ██████  ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("  ██    ██ ██      ████   ██ ██       ██ ██      ██      ██   ██ ██   ██ ", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("  ██    ██ █████   ██ ██  ██ █████     ███       ██      ███████ ██████  ", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("  ██ ▄▄ ██ ██      ██  ██ ██ ██       ██ ██      ██      ██   ██ ██   ██ ", Style::default().fg(Color::Magenta)),
        ]),
        Line::from(vec![
            Span::styled("   ██████  ███████ ██   ████ ███████ ██   ██     ███████ ██   ██ ██████  ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("      ╚═╝                                                                 ", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("                      OMNI-AWARE v1.4.0-INFINITY                          ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        ]),
    ]
}

fn render_header(f: &mut Frame, area: Rect, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(8), Constraint::Length(3)])
        .split(area);

    // ASCII art
    let art = Paragraph::new(render_ascii_art())
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Cyan)));
    f.render_widget(art, chunks[0]);

    // Stats bar
    let uptime = state.start_time.elapsed().as_secs();
    let stats_text = vec![
        Line::from(vec![
            Span::styled("⚡ DOCS: ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled(format!("{}", state.total_documents), Style::default().fg(Color::Green)),
            Span::styled("  │  ", Style::default().fg(Color::DarkGray)),
            Span::styled("📊 MODEL: ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled(
                state.current_model.as_deref().unwrap_or("idle"),
                Style::default().fg(Color::Cyan)
            ),
            Span::styled("  │  ", Style::default().fg(Color::DarkGray)),
            Span::styled("⏱ UPTIME: ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled(format!("{}s", uptime), Style::default().fg(Color::Green)),
        ]),
    ];

    let stats = Paragraph::new(stats_text)
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Blue)));
    f.render_widget(stats, chunks[1]);
}

fn render_expert_panel(f: &mut Frame, area: Rect, state: &AppState) {
    let experts: Vec<ListItem> = state
        .expert_status
        .experts
        .iter()
        .map(|(name, status)| {
            let (symbol, color) = match status.as_str() {
                "idle" => ("○", Color::DarkGray),
                "thinking" => ("◉", Color::Yellow),
                "validated" => ("●", Color::Green),
                "error" => ("✗", Color::Red),
                _ => ("?", Color::White),
            };

            ListItem::new(Line::from(vec![
                Span::styled(symbol, Style::default().fg(color)),
                Span::raw(" "),
                Span::styled(format!("{:12}", name), Style::default().fg(Color::Cyan)),
            ]))
        })
        .collect();

    let list = List::new(experts).block(
        Block::default()
            .borders(Borders::ALL)
            .title("▼ 18-EXPERT MATRIX ▼")
            .title_alignment(Alignment::Center)
            .border_style(Style::default().fg(Color::Magenta)),
    );

    f.render_widget(list, area);
}

fn render_context_panel(f: &mut Frame, area: Rect, state: &AppState) {
    let text = if let Some(ctx) = &state.current_context {
        let mut lines = vec![
            Line::from(vec![
                Span::styled("📚 ACTIVE MEMORY", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(""),
        ];

        for (i, file) in ctx.discovery_files.iter().take(5).enumerate() {
            lines.push(Line::from(vec![
                Span::styled(format!("[{}] ", i + 1), Style::default().fg(Color::Yellow)),
                Span::styled(&file.name, Style::default().fg(Color::Green)),
            ]));
            lines.push(Line::from(vec![
                Span::styled(format!("    └─ {:.3}", file.relevance), Style::default().fg(Color::DarkGray)),
            ]));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("🧠 EXPERTS: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled(
                if ctx.experts.scout_cli { "Scout-CLI " } else { "" },
                Style::default().fg(Color::Green)
            ),
            Span::styled(
                if ctx.experts.qlang { "Q-Lang " } else { "" },
                Style::default().fg(Color::Green)
            ),
            Span::styled(
                if ctx.experts.lagrangian { "Lagrangian" } else { "Fast-Mode" },
                Style::default().fg(Color::Yellow)
            ),
        ]));

        if let Some(time) = ctx.processing_time {
            lines.push(Line::from(vec![
                Span::styled("⚡ TIME: ", Style::default().fg(Color::Cyan)),
                Span::styled(format!("{:.2}s", time), Style::default().fg(Color::Green)),
            ]));
        }

        Text::from(lines)
    } else {
        Text::from(vec![
            Line::from(vec![
                Span::styled("⌛ Awaiting query...", Style::default().fg(Color::DarkGray)),
            ]),
        ])
    };

    let paragraph = Paragraph::new(text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("▼ CONTEXT ▼")
                .title_alignment(Alignment::Center)
                .border_style(Style::default().fg(Color::Blue)),
        )
        .wrap(Wrap { trim: true });

    f.render_widget(paragraph, area);
}

fn render_messages(f: &mut Frame, area: Rect, state: &AppState) {
    let messages: Vec<Line> = state
        .messages
        .iter()
        .flat_map(|msg| {
            msg.lines()
                .map(|line| Line::from(Span::styled(line, Style::default().fg(Color::White))))
                .collect::<Vec<_>>()
        })
        .collect();

    // Calculate total lines after wrapping (approximate)
    let total_lines = messages.len();
    let visible_lines = (area.height as usize).saturating_sub(2); // Subtract borders

    // Show scroll indicators in title
    let scroll_info = if total_lines > visible_lines {
        let current_line = state.response_scroll + 1;
        let last_visible_line = (state.response_scroll + visible_lines).min(total_lines);
        format!(" [Lines {}-{}/{} - Use ↑↓/PgUp/PgDn/Home/End]",
                current_line, last_visible_line, total_lines)
    } else if total_lines > 0 {
        format!(" [All {} lines visible]", total_lines)
    } else {
        String::new()
    };

    let title = format!("▼ RESPONSE{} ▼", scroll_info);

    let paragraph = Paragraph::new(messages)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(title)
                .title_alignment(Alignment::Center)
                .border_style(Style::default().fg(Color::Green)),
        )
        .wrap(Wrap { trim: true })
        .scroll((state.response_scroll as u16, 0));

    f.render_widget(paragraph, area);
}

fn render_log(f: &mut Frame, area: Rect, state: &AppState) {
    let logs: Vec<ListItem> = state
        .log
        .iter()
        .rev()
        .take(area.height as usize - 2)
        .map(|entry| {
            let (prefix, style) = match entry.level {
                LogLevel::Info => ("ℹ", Style::default().fg(Color::Cyan)),
                LogLevel::Success => ("✓", Style::default().fg(Color::Green)),
                LogLevel::Warning => ("⚠", Style::default().fg(Color::Yellow)),
                LogLevel::Error => ("✗", Style::default().fg(Color::Red)),
                LogLevel::Discovery => ("📚", Style::default().fg(Color::Magenta)),
            };

            ListItem::new(Line::from(vec![
                Span::styled(format!("[{}] ", entry.timestamp), Style::default().fg(Color::DarkGray)),
                Span::styled(prefix, style),
                Span::raw(" "),
                Span::styled(&entry.message, style),
            ]))
        })
        .collect();

    let list = List::new(logs).block(
        Block::default()
            .borders(Borders::ALL)
            .title("▼ SYSTEM LOG ▼")
            .title_alignment(Alignment::Center)
            .border_style(Style::default().fg(Color::Yellow)),
    );

    f.render_widget(list, area);
}

fn render_input(f: &mut Frame, area: Rect, state: &AppState) {
    let input_text = if state.is_streaming {
        "⌛ Streaming..."
    } else {
        &state.input
    };

    let input = Paragraph::new(input_text)
        .style(Style::default().fg(Color::White))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("▼ INPUT (Ctrl+C: Quit | Enter: Send) ▼")
                .title_alignment(Alignment::Center)
                .border_style(Style::default().fg(Color::Cyan)),
        );

    f.render_widget(input, area);
}

fn render_processing_gauge(f: &mut Frame, area: Rect, state: &AppState) {
    let ratio = if state.is_streaming { 0.75 } else { 0.0 };

    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title("▼ PROCESSING ▼"))
        .gauge_style(Style::default().fg(Color::Cyan).bg(Color::Black))
        .ratio(ratio);

    f.render_widget(gauge, area);
}

fn draw_ui(f: &mut Frame, state: &AppState) {
    let size = f.area();

    // Main layout: header + body
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(11), // Header
            Constraint::Min(10),     // Body
            Constraint::Length(3),   // Input
            Constraint::Length(3),   // Gauge
        ])
        .split(size);

    render_header(f, main_chunks[0], state);

    // Body: left (messages + log) + right (experts + context)
    let body_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(main_chunks[1]);

    // Left side: messages + log
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(body_chunks[0]);

    render_messages(f, left_chunks[0], state);
    render_log(f, left_chunks[1], state);

    // Right side: experts + context
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(body_chunks[1]);

    render_expert_panel(f, right_chunks[0], state);
    render_context_panel(f, right_chunks[1], state);

    render_input(f, main_chunks[2], state);
    render_processing_gauge(f, main_chunks[3], state);
}

// ============================================================================
// MAIN
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture, EnableBracketedPaste)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let state = Arc::new(Mutex::new(AppState::new()));
    let (tx, mut rx) = mpsc::channel::<UIEvent>(100);

    // Spawn event handler
    let state_clone = Arc::clone(&state);
    tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            let mut state = state_clone.lock().await;

            match event {
                UIEvent::Message(content) => {
                    if state.messages.is_empty() {
                        state.messages.push(content);
                    } else {
                        let last_idx = state.messages.len() - 1;
                        state.messages[last_idx].push_str(&content);
                    }
                }
                UIEvent::Context(ctx) => {
                    state.current_context = Some(ctx.clone());
                    state.processing_time = ctx.processing_time.unwrap_or(0.0);

                    for file in &ctx.discovery_files {
                        state.add_log(
                            format!("📚 Using: {} ({:.3})", file.name, file.relevance),
                            LogLevel::Discovery,
                        );
                    }
                }
                UIEvent::Model(model) => {
                    state.current_model = Some(model.clone());
                    state.add_log(format!("🤖 Model: {}", model), LogLevel::Info);
                }
                UIEvent::ToolCall(tool) => {
                    state.add_log(
                        format!("🔬 Consulting {} for expert guidance...",
                            if tool == "scout" { "Scout 17B" } else { &tool }
                        ),
                        LogLevel::Info
                    );
                }
                UIEvent::ToolResult(result) => {
                    state.add_log(
                        format!("✓ Expert consultation complete: {}", result),
                        LogLevel::Success
                    );
                }
                UIEvent::Done => {
                    state.is_streaming = false;
                    state.add_log("✓ Response complete".to_string(), LogLevel::Success);
                }
                UIEvent::Error(err) => {
                    state.add_log(format!("✗ Error: {}", err), LogLevel::Error);
                    state.is_streaming = false;
                }
                UIEvent::ExpertStatus(status) => {
                    state.expert_status = status;
                }
            }
        }
    });

    // Main loop
    let tick_rate = Duration::from_millis(100);
    let mut last_tick = Instant::now();

    loop {
        // Draw UI
        {
            let state = state.lock().await;
            terminal.draw(|f| draw_ui(f, &state))?;
        }

        // Handle input
        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if event::poll(timeout)? {
            let event = event::read()?;
            let mut state = state.lock().await;

            match event {
                Event::Paste(p) if !state.is_streaming => {
                    state.input.push_str(&p);
                }
                Event::Key(key) => match key.code {
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        break;
                    }
                    KeyCode::Up if state.is_streaming || !state.messages.is_empty() => {
                        // Scroll up in response view (line by line)
                        if state.response_scroll > 0 {
                            state.response_scroll = state.response_scroll.saturating_sub(1);
                            let scroll_val = state.response_scroll;
                            state.add_log(
                                format!("↑ Scrolled to line {}", scroll_val),
                                LogLevel::Info
                            );
                        }
                    }
                    KeyCode::Down if state.is_streaming || !state.messages.is_empty() => {
                        // Scroll down in response view (line by line)
                        let total_lines: usize = state.messages.iter()
                            .map(|msg| msg.lines().count())
                            .sum();
                        let max_scroll = total_lines.saturating_sub(20);

                        if state.response_scroll < max_scroll {
                            state.response_scroll += 1;
                            let scroll_val = state.response_scroll;
                            state.add_log(
                                format!("↓ Scrolled to line {} (max: {})", scroll_val, max_scroll),
                                LogLevel::Info
                            );
                        }
                    }
                    KeyCode::PageUp if state.is_streaming || !state.messages.is_empty() => {
                        // Scroll up by page (10 lines)
                        state.response_scroll = state.response_scroll.saturating_sub(10);
                    }
                    KeyCode::PageDown if state.is_streaming || !state.messages.is_empty() => {
                        // Scroll down by page (10 lines)
                        let total_lines: usize = state.messages.iter()
                            .map(|msg| msg.lines().count())
                            .sum();
                        let max_scroll = total_lines.saturating_sub(20);

                        state.response_scroll = (state.response_scroll + 10).min(max_scroll);
                    }
                    KeyCode::Home if state.is_streaming || !state.messages.is_empty() => {
                        // Scroll to top
                        state.response_scroll = 0;
                    }
                    KeyCode::End if state.is_streaming || !state.messages.is_empty() => {
                        // Scroll to bottom
                        let total_lines: usize = state.messages.iter()
                            .map(|msg| msg.lines().count())
                            .sum();
                        state.response_scroll = total_lines.saturating_sub(20);
                    }
                    KeyCode::Enter if !state.is_streaming && !state.input.is_empty() => {
                        let query = state.input.clone();
                        state.input.clear();
                        state.messages.clear();
                        state.messages.push(String::new());
                        state.is_streaming = true;
                        state.current_context = None;
                        state.response_scroll = 0;  // Reset scroll on new query

                        state.add_log(format!("→ Query: {}", query), LogLevel::Info);

                        let tx_clone = tx.clone();
                        tokio::spawn(async move {
                            if let Err(e) = send_query(query, tx_clone.clone()).await {
                                let _ = tx_clone.send(UIEvent::Error(e.to_string())).await;
                            }
                        });
                    }
                    KeyCode::Char(c) if !state.is_streaming => {
                        state.input.push(c);
                    }
                    KeyCode::Backspace if !state.is_streaming => {
                        state.input.pop();
                    }
                    _ => {}
                },
                _ => {}
            }
        }

        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture,
        DisableBracketedPaste
    )?;
    terminal.show_cursor()?;

    Ok(())
}
