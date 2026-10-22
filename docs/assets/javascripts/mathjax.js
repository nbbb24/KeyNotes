window.MathJax = {
  tex: { inlineMath: [["\\(", "\\)"]], displayMath: [["\\[", "\\]"]], processEscapes: true },
  options: { ignoreHtmlClass: ".*|", processHtmlClass: "arithmatex" },
  startup: { pageReady: () => MathJax.startup.defaultPageReady() }
};
document$.subscribe(() => {
  if (window.MathJax.typesetPromise) {
    MathJax.startup.promise = MathJax.startup.promise.then(() => {
      MathJax.typesetClear();
      return MathJax.typesetPromise();
    });
  }
});
