export default {
  arrowParens: "always",
  bracketSameLine: true,
  bracketSpacing: true,
  htmlWhitespaceSensitivity: "ignore",
  overrides: [{ files: "*.svelte", options: { parser: "svelte" } }],
  plugins: ["prettier-plugin-svelte"],
  printWidth: 100,
  semi: true,
  singleQuote: false,
  tabWidth: 2,
  trailingComma: "none",
  useTabs: false
};
