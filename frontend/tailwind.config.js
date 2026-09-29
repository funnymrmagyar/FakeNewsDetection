/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0B0F14",
        panel: "#121821",
        panelBorder: "#1E2733",
        real: "#33D6C0",
        fake: "#FF6B4A",
        muted: "#8894A3",
      },
      fontFamily: {
        display: ['"Space Grotesk"', "sans-serif"],
        mono: ['"IBM Plex Mono"', "monospace"],
        body: ["Inter", "sans-serif"],
      },
    },
  },
  plugins: [],
}