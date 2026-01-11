/**
 * Application-wide constants and configuration
 */
export const config = {
  // Base URL
  baseUrl: "https://qenex.ai",

  // GitHub
  github: {
    repoUrl: "https://github.com/abdulrahman305/qenex-lab",
    starsFormatted: {
      compact: "50K",
      full: "50,000",
    },
  },

  // Social links
  social: {
    twitter: "https://x.com/qenex_lab",
    discord: "https://discord.gg/qenex-lab",
  },

  // Static stats (used on landing page)
  stats: {
    contributors: "500",
    commits: "6,500",
    monthlyUsers: "650,000",
  },
} as const
