
document.addEventListener("DOMContentLoaded", () => {
  const button = document.getElementById("themeToggleBtn");
  const storageKey = "enervision-theme";

  function applyTheme(theme) {
    document.body.classList.toggle("light-theme", theme === "light");
    document.body.classList.toggle("dark-theme", theme === "dark");

    if (button) {
      button.textContent =
        theme === "dark" ? "☀️ Light Mode" : "🌙 Dark Mode";

      button.setAttribute(
        "aria-label",
        theme === "dark" ? "Switch to light theme" : "Switch to dark theme"
      );
    }
  }

  let theme;

  try {
    theme = localStorage.getItem(storageKey);
  } catch (_) {}

  if (theme !== "light" && theme !== "dark") {
    theme = window.matchMedia?.("(prefers-color-scheme: light)").matches
      ? "light"
      : "dark";
  }

  applyTheme(theme);

  button?.addEventListener("click", () => {
    const nextTheme = document.body.classList.contains("light-theme")
      ? "dark"
      : "light";

    applyTheme(nextTheme);

    try {
      localStorage.setItem(storageKey, nextTheme);
    } catch (_) {}
  });
});
