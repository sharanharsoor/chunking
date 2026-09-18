(function () {
  var page = (location.pathname.replace(/\/+$/, "").split("/").pop() || "index.html");
  if (page === "docs" || page.indexOf(".html") === -1) page = "index.html";

  var groups = [
    ["Start", [
      ["./", "Overview", "index.html"],
      ["lab.html", "The lab"],
      ["install.html", "Install"],
      ["cli.html", "CLI only"]
    ]],
    ["The library can, the tab cannot", [
      ["compare.html", "N-way compare"],
      ["custom.html", "Bring your own"]
    ]],
    ["Worked paths", [
      ["json.html", "JSON"],
      ["ml.html", "ML and embeddings"],
      ["config.html", "Config and scale"],
      ["logging.html", "Logs and debug"]
    ]],
    ["Reference", [
      ["strategies.html", "Strategies"],
      ["python.html", "Python API"],
      ["formats.html", "Formats"]
    ]]
  ];

  var nav = document.querySelector("nav.side");
  if (nav) {
    var html = "";
    groups.forEach(function (g) {
      html += "<p>" + g[0] + "</p>";
      g[1].forEach(function (item) {
        var href = item[0];
        var label = item[1];
        var match = item[2] || href;
        var here = page === match || (page === "index.html" && href === "./index.html");
        html += '<a' + (here ? ' class="here"' : "") + ' href="' + href + '">' + label + "</a>";
      });
    });
    nav.innerHTML = html;
  }

  var theme = document.getElementById("theme");
  if (theme) {
    theme.addEventListener("click", function () {
      var next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      localStorage.setItem("theme", next);
    });
  }

  document.querySelectorAll("pre").forEach(function (pre) {
    if (pre.parentNode && pre.parentNode.classList.contains("code")) return;
    var wrap = document.createElement("div");
    wrap.className = "code";
    pre.parentNode.insertBefore(wrap, pre);
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy";
    btn.textContent = "Copy";
    wrap.appendChild(btn);
    wrap.appendChild(pre);
    btn.addEventListener("click", function () {
      var text = pre.innerText || pre.textContent || "";
      function done() {
        btn.textContent = "Copied";
        setTimeout(function () { btn.textContent = "Copy"; }, 1400);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done).catch(function () {
          fallback(text, done);
        });
      } else {
        fallback(text, done);
      }
    });
  });

  function fallback(text, done) {
    var ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); } catch (e) {}
    document.body.removeChild(ta);
    done();
  }
})();
