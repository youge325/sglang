/**
 * Language Switcher for SGLang Documentation
 *
 * Provides smart language switching that preserves the current page path
 * when switching between English and Chinese documentation.
 */
document.addEventListener("DOMContentLoaded", function () {
  // Find all language switcher links in the announcement bar
  const announceBar = document.querySelector(".bd-header-announcement");
  if (!announceBar) return;

  const links = announceBar.querySelectorAll("a[href]");
  links.forEach(function (link) {
    link.addEventListener("click", function (e) {
      e.preventDefault();
      const targetHref = link.getAttribute("href");

      // Determine the target language from the href
      let targetLang = "en";
      if (targetHref.includes("/zh_CN/")) {
        targetLang = "zh_CN";
      }

      // Get current path and determine current language
      const currentPath = window.location.pathname;
      let currentLang = "en";
      if (currentPath.includes("/zh_CN/")) {
        currentLang = "zh_CN";
      }

      // If already on the target language, do nothing
      if (currentLang === targetLang) return;

      // Build new path by replacing the language segment
      let newPath;
      if (currentLang === "en") {
        newPath = currentPath.replace(/\/en\//, "/" + targetLang + "/");
      } else {
        newPath = currentPath.replace(
          /\/zh_CN\//,
          "/" + targetLang + "/"
        );
      }

      // Navigate to the new path, fallback to index if page doesn't exist
      window.location.href = newPath;
    });
  });
});
