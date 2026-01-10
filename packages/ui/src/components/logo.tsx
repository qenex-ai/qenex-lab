export const Mark = (props: { class?: string }) => {
  return (
    <svg
      data-component="logo-mark"
      classList={{ [props.class ?? ""]: !!props.class }}
      viewBox="0 0 20 20"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path data-slot="qenex-mark-bg" d="M10 2L18 6V14L10 18L2 14V6L10 2Z" fill="var(--icon-weak-base)" />
      <path data-slot="qenex-mark-fg" d="M10 5L15 7.5V12.5L10 15L5 12.5V7.5L10 5Z" fill="var(--icon-strong-base)" />
    </svg>
  )
}

export const Logo = (props: { class?: string }) => {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 240 42"
      fill="none"
      classList={{ [props.class ?? ""]: !!props.class }}
    >
      <g>
        {/* Q */}
        <path d="M6 6H24V30H30V36H20L16 30H6V6ZM12 12V24H18V12H12Z" fill="var(--icon-strong-base)" />
        {/* E */}
        <path d="M36 6H60V12H42V18H54V24H42V30H60V36H36V6Z" fill="var(--icon-strong-base)" />
        {/* N */}
        <path d="M66 6H72L84 24V6H90V36H84L72 18V36H66V6Z" fill="var(--icon-strong-base)" />
        {/* E */}
        <path d="M96 6H120V12H102V18H114V24H102V30H120V36H96V6Z" fill="var(--icon-strong-base)" />
        {/* X */}
        <path
          d="M126 6H134L141 18L148 6H156L145 21L156 36H148L141 24L134 36H126L137 21L126 6Z"
          fill="var(--icon-strong-base)"
        />

        {/* LAB (Thinner/Light) */}
        <path d="M174 6H180V30H192V36H174V6Z" fill="var(--icon-weak-base)" />
        <path
          d="M198 36H204L210 6H216L222 36H228L225 24H195L198 36ZM210 12L213 24H207L210 12Z"
          fill="var(--icon-weak-base)"
        />
        <path
          d="M234 6H246C252 6 252 12 246 12H240V15C246 15 252 18 252 24C252 30 246 36 234 36H234V6ZM240 12V12H240V12ZM240 21V30H243C245 30 246 29 246 25C246 22 245 21 243 21H240Z"
          fill="var(--icon-weak-base)"
        />
      </g>
    </svg>
  )
}
