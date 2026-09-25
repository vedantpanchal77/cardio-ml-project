/* Front end for the cardiovascular disease prediction project.
 * Every number on this page comes from the Flask API in app.py — nothing is
 * computed here except drawing charts and formatting text.
 */
(function () {
  "use strict";

  function fmtPct(v, d) { return (v * 100).toFixed(d == null ? 1 : d) + "%"; }
  function el(tag, attrs, text) {
    var n = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (var a in attrs) n.setAttribute(a, attrs[a]);
    if (text != null) n.textContent = text;
    return n;
  }
  function clear(node) { if (node) { while (node.firstChild) node.removeChild(node.firstChild); } }
  function statBox(v, l, n) {
    return '<div class="stat"><span class="v mono">' + v + '</span><span class="l">' + l + "</span>" +
      (n ? '<span class="n">' + n + "</span>" : "") + "</div>";
  }

  function setTxt(id, txt) { var e = document.getElementById(id); if (e) e.textContent = txt; }
  function setHtml(id, html) { var e = document.getElementById(id); if (e) e.innerHTML = html; }
  function removeClass(id, cls) { var e = document.getElementById(id); if (e) e.classList.remove(cls); }

  async function api(path, opts) {
    var res = await fetch(path, opts);
    var data = await res.json().catch(function () { return {}; });
    if (!res.ok) {
      var err = new Error(data.error || ("request to " + path + " failed"));
      err.details = data.details;
      err.status = res.status;
      throw err;
    }
    return data;
  }
  function showApiError(message) {
    var box = document.getElementById("apiError");
    if (box) {
      box.textContent = message;
      box.hidden = false;
    }
  }

  /* ---------------- ECG hero path (decoration only, no data) ---------------- */
  (function () {
    var ecg = document.getElementById("ecgPath");
    if (!ecg) return;
    var pts = [], x = 0, W = 1240, base = 104;
    var beat = [[26, 0], [8, 9], [8, 0], [8, 0], [4, -7], [6, 56], [6, -15], [8, 0], [12, 0], [14, 17], [16, 0], [22, 0]];
    pts.push([0, base]);
    while (x < W) { for (var i = 0; i < beat.length; i++) { x += beat[i][0]; pts.push([x, base - beat[i][1]]); } }
    var d = "M " + pts.map(function (p) { return p[0].toFixed(1) + " " + p[1].toFixed(1); }).join(" L ");
    ecg.setAttribute("d", d);
  })();

  /* ---------------- charts ---------------- */
  function drawLoss(history, epochs) {
    var svg = document.getElementById("lossChart");
    if (!svg) return;
    clear(svg);
    var L = 46, R = 12, T = 12, Bm = 30, Wd = 440, Ht = 240;
    var pw = Wd - L - R, ph = Ht - T - Bm;
    var lo = Math.min.apply(null, history) - 0.005;
    var hi = Math.max(history[0], history[0] * 1.02);
    var yv = function (v) { return T + ph - (v - lo) / (hi - lo) * ph; };
    var xv = function (i) { return L + (i / (epochs - 1)) * pw; };
    [0, .25, .5, .75, 1].forEach(function (f) {
      var v = lo + f * (hi - lo), yy = yv(v);
      svg.appendChild(el("line", { x1: L, x2: L + pw, y1: yy, y2: yy, class: "gridline" }));
      svg.appendChild(el("text", { x: L - 8, y: yy + 3.5, class: "axtxt", "text-anchor": "end" }, v.toFixed(2)));
    });
    svg.appendChild(el("line", { x1: L, x2: L + pw, y1: T + ph, y2: T + ph, class: "ax" }));
    [0, .25, .5, .75, 1].forEach(function (f) {
      var e = Math.round(f * epochs), xx = L + f * pw;
      svg.appendChild(el("text", { x: xx, y: T + ph + 17, class: "axtxt", "text-anchor": "middle" }, e));
    });
    svg.appendChild(el("text", { x: L + pw / 2, y: Ht - 2, class: "axtxt", "text-anchor": "middle" }, "epoch"));
    var d = "M " + history.map(function (v, i) { return xv(i).toFixed(1) + " " + yv(v).toFixed(1); }).join(" L ");
    svg.appendChild(el("path", { d: d, fill: "none", stroke: "var(--trace)", "stroke-width": 2, "stroke-linejoin": "round" }));
    var last = history[history.length - 1];
    svg.appendChild(el("circle", { cx: xv(history.length - 1), cy: yv(last), r: 3, fill: "var(--trace)" }));
  }

  function drawRoc(fpr, tpr, auc) {
    var svg = document.getElementById("rocChart");
    if (!svg) return;
    clear(svg);
    var L = 44, R = 14, T = 14, Bm = 36, Wd = 440, Ht = 300;
    var pw = Wd - L - R, ph = Ht - T - Bm;
    var xv = function (v) { return L + v * pw; }, yv = function (v) { return T + ph - v * ph; };
    [0, .25, .5, .75, 1].forEach(function (f) {
      svg.appendChild(el("line", { x1: L, x2: L + pw, y1: yv(f), y2: yv(f), class: "gridline" }));
      svg.appendChild(el("text", { x: L - 8, y: yv(f) + 3.5, class: "axtxt", "text-anchor": "end" }, f.toFixed(2)));
      svg.appendChild(el("text", { x: xv(f), y: T + ph + 17, class: "axtxt", "text-anchor": "middle" }, f.toFixed(2)));
    });
    svg.appendChild(el("line", { x1: L, y1: yv(0), x2: L + pw, y2: yv(1), stroke: "var(--rule)", "stroke-width": 1, "stroke-dasharray": "4 4" }));
    var d = "M " + fpr.map(function (f, i) { return xv(f).toFixed(1) + " " + yv(tpr[i]).toFixed(1); }).join(" L ");
    svg.appendChild(el("path", { d: d + " L " + xv(1) + " " + yv(0) + " Z", fill: "var(--trace-soft)", stroke: "none" }));
    svg.appendChild(el("path", { d: d, fill: "none", stroke: "var(--trace)", "stroke-width": 2.2, "stroke-linejoin": "round" }));
    svg.appendChild(el("text", { x: L + pw / 2, y: Ht - 4, class: "axtxt", "text-anchor": "middle" }, "false positive rate"));
    svg.appendChild(el("text", { x: 12, y: T + ph / 2, class: "axtxt", "text-anchor": "middle", transform: "rotate(-90 12 " + (T + ph / 2) + ")" }, "true positive rate"));
    ROC.marker = el("circle", { r: 4.5, fill: "var(--surface)", stroke: "var(--trace)", "stroke-width": 2.5 });
    svg.appendChild(ROC.marker);
    ROC.xv = xv; ROC.yv = yv;
  }
  var ROC = {};

  function drawCoefs(features) {
    var host = document.getElementById("coefChart");
    if (!host) return;
    host.innerHTML = "";
    var order = features.slice().sort(function (a, b) { return Math.abs(b.weight) - Math.abs(a.weight); });
    var max = Math.max.apply(null, order.map(function (o) { return Math.abs(o.weight); }));
    order.forEach(function (o) {
      var row = document.createElement("div"); row.className = "bar-row";
      var lab = document.createElement("span"); lab.textContent = o.label; lab.style.color = "var(--ink-2)";
      var track = document.createElement("div"); track.className = "bar-track";
      var mid = document.createElement("div"); mid.className = "bar-mid";
      var fill = document.createElement("div"); fill.className = "bar-fill";
      var frac = Math.abs(o.weight) / max * 0.5;
      if (o.weight >= 0) { fill.style.left = "50%"; fill.style.background = "var(--trace)"; }
      else { fill.style.right = "50%"; fill.style.background = "var(--well)"; }
      fill.style.width = (frac * 100) + "%";
      track.appendChild(mid); track.appendChild(fill);
      var val = document.createElement("span"); val.className = "bar-val";
      val.textContent = (o.weight >= 0 ? "+" : "") + o.weight.toFixed(3);
      row.appendChild(lab); row.appendChild(track); row.appendChild(val);
      host.appendChild(row);
    });
  }

  /* ---------------- state ---------------- */
  var MODEL = null;     // response of /api/model
  var form = { age: 55, gender: 1, height: 165, weight: 74, ap_hi: 120, ap_lo: 80, cholesterol: 1, gluc: 1, smoke: 0, alco: 0, active: 1 };
  var predictSeq = 0;   // guards against out-of-order responses when sliders move fast

  /* ---------------- sections ---------------- */
  async function loadSummary() {
    var s = await api("/api/summary");
    var ds = s.dataset, sp = s.split;
    setTxt("hRows", ds.clean_rows.toLocaleString());
    removeClass("hRows", "pending");
    setTxt("trainRowsInline", sp.train_rows.toLocaleString());
    setTxt("testRowsInline", sp.test_rows.toLocaleString());
    setHtml("dataStats",
      statBox(ds.raw_rows.toLocaleString(), "Records in the file", "raw rows before any filtering") +
      statBox(ds.clean_rows.toLocaleString(), "Survive cleaning", fmtPct(ds.removed_pct) + " removed as physically impossible") +
      statBox(sp.train_rows.toLocaleString() + " / " + sp.test_rows.toLocaleString(), "Train / test split", "stratified 80/20 split") +
      statBox("12", "Features used", "eleven measured, plus BMI derived here") +
      statBox(fmtPct(ds.positive_rate, 1), "Have the disease", "the classes are almost perfectly balanced") +
      statBox(ds.mean_age_years.toFixed(1), "Mean age (years)", "stored in the file as days since birth") +
      statBox(ds.mean_bmi.toFixed(1), "Mean BMI", "weight ÷ height², computed at load") +
      statBox(ds.mean_systolic.toFixed(0), "Mean systolic", "mmHg, the strongest single signal")
    );

    var body = ds.rules.map(function (r) {
      var range = r.low == null ? "must hold" : r.low + "–" + r.high + (r.unit ? " " + r.unit : "");
      return "<tr><td>" + r.label + "</td><td class=\"num\">" + range + "</td><td class=\"num\">" +
        r.rows_failing.toLocaleString() + "</td><td>" + r.why + "</td></tr>";
    }).join("");
    setHtml("rulesBody", body);
    setTxt("dataNote",
      ds.removed_rows.toLocaleString() + " of " + ds.raw_rows.toLocaleString() + " rows (" + fmtPct(ds.removed_pct) +
      ") fail at least one rule and are removed, leaving " + ds.clean_rows.toLocaleString() +
      ". A row can fail more than one rule, so the counts above overlap."
    );
  }

  async function loadModel() {
    MODEL = await api("/api/model");
    removeClass("hAcc", "pending");
    removeClass("hAuc", "pending");
    setTxt("hTime", MODEL.training_seconds.toFixed(2) + " s");
    removeClass("hTime", "pending");
    setTxt("trainStatus",
      "Converged after " + MODEL.hyperparameters.epochs + " epochs. Final training loss " +
      MODEL.loss_history[MODEL.loss_history.length - 1].toFixed(4) + ", trained in " +
      MODEL.training_seconds.toFixed(2) + " s on the server."
    );
    setHtml("hyperBody",
      "<tr><td>Learning rate α</td><td class=\"num\">" + MODEL.hyperparameters.learning_rate.toFixed(2) + "</td></tr>" +
      "<tr><td>Epochs</td><td class=\"num\">" + MODEL.hyperparameters.epochs + "</td></tr>" +
      "<tr><td>L2 strength λ</td><td class=\"num\">" + MODEL.hyperparameters.l2.toFixed(2) + "</td></tr>" +
      "<tr><td>Train / test split</td><td class=\"num\">" + MODEL.train_rows.toLocaleString() + " / " + MODEL.test_rows.toLocaleString() + "</td></tr>" +
      "<tr><td>Cross-val accuracy</td><td class=\"num\">" + fmtPct(MODEL.cross_validation.accuracy_mean) + " ± " + (MODEL.cross_validation.accuracy_std * 100).toFixed(2) + "pt</td></tr>"
    );
    drawLoss(MODEL.loss_history, MODEL.hyperparameters.epochs);
    drawCoefs(MODEL.features);
    var hi = MODEL.features.find(function (f) { return f.key === "ap_hi"; });
    var age = MODEL.features.find(function (f) { return f.key === "age_years"; });
    if (hi && age) {
      setTxt("coefNote",
        "Red pushes the prediction toward disease, teal away from it. Systolic pressure at " + hi.weight.toFixed(3) +
        " carries roughly " + Math.abs(hi.weight / age.weight).toFixed(1) + " times the weight of age, and the intercept sits at " + MODEL.bias.toFixed(3) + "."
      );
    }
  }

  async function loadRoc() {
    var r = await api("/api/roc");
    drawRoc(r.fpr, r.tpr, r.auc);
    setTxt("hAuc", r.auc.toFixed(3));
  }

  async function loadEvaluate(threshold) {
    var m = await api("/api/evaluate?threshold=" + threshold.toFixed(2));
    setTxt("thrVal", threshold.toFixed(2));
    setTxt("hAcc", fmtPct(m.accuracy));
    setHtml("metricStats",
      statBox(fmtPct(m.accuracy), "Accuracy", "share of the " + m.n.toLocaleString() + " test patients labelled correctly") +
      statBox(fmtPct(m.precision), "Precision", "of those flagged, how many really have disease") +
      statBox(fmtPct(m.recall), "Recall", "of those with disease, how many were caught") +
      statBox(fmtPct(m.f1), "F1 score", "harmonic mean of precision and recall") +
      statBox(m.auc.toFixed(3), "ROC AUC", "threshold-independent ranking quality") +
      statBox(fmtPct(m.specificity), "Specificity", "of the healthy, how many were left alone")
    );
    setHtml("cm",
      '<div class="corner"></div>' +
      '<div class="axis" style="justify-content:center">predicted healthy</div>' +
      '<div class="axis" style="justify-content:center">predicted disease</div>' +
      '<div class="axis">actually<br>healthy</div>' +
      '<div class="cell hit"><b>' + m.tn + '</b><span>true negative</span></div>' +
      '<div class="cell miss"><b>' + m.fp + '</b><span>false alarm</span></div>' +
      '<div class="axis">actually<br>diseased</div>' +
      '<div class="cell miss"><b>' + m.fn + '</b><span>missed case</span></div>' +
      '<div class="cell hit"><b>' + m.tp + '</b><span>true positive</span></div>'
    );
    setTxt("cmNote",
      "At this threshold the model misses " + m.fn + " patients who have the disease and raises " + m.fp +
      " false alarms. Dropping the threshold trades the first number for the second."
    );
    if (ROC.marker) {
      ROC.marker.setAttribute("cx", ROC.xv(1 - m.specificity));
      ROC.marker.setAttribute("cy", ROC.yv(m.recall));
    }
  }

  async function loadCompare() {
    var c = await api("/api/compare");
    var body = c.models.map(function (r) {
      return '<tr' + (r.key === c.best_accuracy_key ? ' class="best"' : '') + '><td>' + r.name +
        '<br><span class="note" style="margin:0">' + r.note + '</span></td>' +
        '<td class="num">' + fmtPct(r.accuracy) + '</td><td class="num">' + fmtPct(r.precision) + '</td>' +
        '<td class="num">' + fmtPct(r.recall) + '</td><td class="num">' + fmtPct(r.f1) + '</td>' +
        '<td class="num">' + r.auc.toFixed(3) + '</td><td class="num">' + r.train_seconds.toFixed(2) + 's</td></tr>';
    }).join("");
    var tbody = document.querySelector("#cmpTable tbody");
    if (tbody) tbody.innerHTML = body;
    var sk = c.sklearn_check;
    setTxt("cmpNote",
      "The scratch model and scikit-learn's logistic regression agree to within " +
      sk.max_abs_weight_diff.toFixed(4) + " on every weight and " + (sk.max_abs_probability_diff * 100).toFixed(2) +
      " percentage points on every test-set probability — evidence the from-scratch version is implemented correctly."
    );
    var best = c.models.reduce(function (a, b) { return b.accuracy > a.accuracy ? b : a; });
    var lr = c.models.find(function (m) { return m.key === "logreg_scratch"; });
    setTxt("noteCeiling",
      "That ceiling shows here: " + best.name.toLowerCase() + " reaches " + fmtPct(best.accuracy) +
      ", about " + ((best.accuracy - lr.accuracy) * 100).toFixed(1) + " points above logistic regression's " + fmtPct(lr.accuracy) + "."
    );
    var accs = c.models.map(function (m) { return m.accuracy; });
    setTxt("noteAccuracy",
      "Accuracy sits between " + fmtPct(Math.min.apply(null, accs), 0) + " and " + fmtPct(Math.max.apply(null, accs), 0)
    );
  }

  /* ---------------- predictor ---------------- */
  async function predict() {
    var thrElem = document.getElementById("thr");
    if (!thrElem && !document.getElementById("prob")) return; // Predictor section not on this page
    var seq = ++predictSeq;
    var body = Object.assign({ threshold: thrElem ? Number(thrElem.value) / 100 : 0.5 }, form);
    var data;
    try {
      data = await api("/api/predict", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
    } catch (e) {
      showApiError(e.message + (e.details ? ": " + JSON.stringify(e.details) : ""));
      return;
    }
    if (seq !== predictSeq) return;

    setTxt("prob", (data.probability * 100).toFixed(1) + "%");
    var band = document.getElementById("band");
    if (band) {
      band.textContent = data.band.text;
      var colors = { low: ["var(--well-soft)", "var(--well)"], mid: ["var(--surface-2)", "var(--ink-2)"], high: ["var(--trace-soft)", "var(--trace)"] };
      band.style.background = colors[data.band.key][0];
      band.style.color = colors[data.band.key][1];
    }
    setTxt("bmiNote", "BMI works out at " + data.bmi.toFixed(1) + ", derived from height and weight rather than read from a file.");

    var rangeNote = document.getElementById("rangeNote");
    if (rangeNote) {
      if (data.out_of_range.length) {
        rangeNote.hidden = false;
        rangeNote.textContent = "Outside the range the model was trained on: " +
          data.out_of_range.map(function (r) { return r.label.toLowerCase(); }).join(", ") +
          ". The prediction is an extrapolation here.";
      } else {
        rangeNote.hidden = true;
      }
    }

    var top = data.contributions.slice(0, 6), max = Math.abs(top[0].contribution) || 1, html = "";
    top.forEach(function (o) {
      var frac = Math.abs(o.contribution) / max * 0.5;
      html += '<div class="bar-row"><span style="color:var(--ink-2)">' + o.label + "</span>" +
        '<div class="bar-track"><div class="bar-mid"></div><div class="bar-fill" style="' +
        (o.contribution >= 0 ? "left:50%;background:var(--trace);" : "right:50%;background:var(--well);") +
        "width:" + (frac * 100) + '%"></div></div>' +
        '<span class="bar-val">' + (o.contribution >= 0 ? "+" : "") + o.contribution.toFixed(2) + "</span></div>";
    });
    setHtml("contrib", html);
    setTxt("logitNote", "These add to a log-odds of " + data.logit.toFixed(2) + ", which the sigmoid turns into " + (data.probability * 100).toFixed(1) + "%.");

    setHtml("others", data.other_models.map(function (o) {
      return '<div class="other-row"><span>' + o.name + '</span><span class="p">' + (o.probability * 100).toFixed(1) + '%</span></div>';
    }).join(""));
  }

  function bindRange(id, key, unit) {
    var input = document.getElementById(id), out = document.getElementById("v_" + id.slice(2));
    if (!input) return;
    input.addEventListener("input", function () {
      form[key] = Number(input.value); 
      if (out) out.textContent = input.value + " " + unit;
      if (key === "ap_hi" && form.ap_lo >= form.ap_hi) {
        form.ap_lo = form.ap_hi - 10;
        var flo = document.getElementById("f_lo");
        if (flo) flo.value = form.ap_lo;
        setTxt("v_lo", form.ap_lo + " mmHg");
      }
      if (key === "ap_lo" && form.ap_lo >= form.ap_hi) {
        form.ap_hi = form.ap_lo + 10;
        var fhi = document.getElementById("f_hi");
        if (fhi) fhi.value = form.ap_hi;
        setTxt("v_hi", form.ap_hi + " mmHg");
      }
      predict();
    });
  }
  bindRange("f_age", "age", "years");
  bindRange("f_height", "height", "cm");
  bindRange("f_weight", "weight", "kg");
  bindRange("f_hi", "ap_hi", "mmHg");
  bindRange("f_lo", "ap_lo", "mmHg");

  function bindSeg(id, key) {
    var host = document.getElementById(id);
    if (!host) return;
    host.addEventListener("click", function (e) {
      var btn = e.target.closest("button"); if (!btn) return;
      Array.prototype.forEach.call(host.children, function (b) { b.setAttribute("aria-pressed", String(b === btn)); });
      form[key] = Number(btn.dataset.v); predict();
    });
  }
  bindSeg("f_gender", "gender"); bindSeg("f_chol", "cholesterol"); bindSeg("f_gluc", "gluc");

  var fLife = document.getElementById("f_life");
  if (fLife) {
    fLife.addEventListener("click", function (e) {
      var btn = e.target.closest("button"); if (!btn) return;
      var on = btn.getAttribute("aria-pressed") !== "true";
      btn.setAttribute("aria-pressed", String(on));
      form[btn.dataset.k] = on ? 1 : 0; predict();
    });
  }

  var thrInput = document.getElementById("thr");
  var thrTimer = null;
  if (thrInput) {
    thrInput.addEventListener("input", function () {
      setTxt("thrVal", (Number(thrInput.value) / 100).toFixed(2));
      clearTimeout(thrTimer);
      thrTimer = setTimeout(function () {
        loadEvaluate(Number(thrInput.value) / 100).catch(function (e) { showApiError(e.message); });
        predict();
      }, 60);
    });
  }

  /* ---------------- theme & presets ---------------- */
  (function initTheme() {
    var saved = localStorage.getItem("theme");
    var prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    var current = saved || (prefersDark ? "dark" : "light");
    function setTheme(t) {
      document.documentElement.setAttribute("data-theme", t);
      localStorage.setItem("theme", t);
      var btn = document.getElementById("themeToggle");
      if (btn) {
        setTxt("themeIcon", t === "dark" ? "☀️" : "🌙");
        setTxt("themeText", t === "dark" ? "Light mode" : "Dark mode");
      }
    }
    setTheme(current);
    var btn = document.getElementById("themeToggle");
    if (btn) {
      btn.addEventListener("click", function () {
        var next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
        setTheme(next);
      });
    }
  })();

  var PRESETS = {
    healthy: { age: 35, gender: 1, height: 168, weight: 62, ap_hi: 110, ap_lo: 70, cholesterol: 1, gluc: 1, smoke: 0, alco: 0, active: 1 },
    borderline: { age: 52, gender: 2, height: 175, weight: 82, ap_hi: 135, ap_lo: 88, cholesterol: 2, gluc: 1, smoke: 0, alco: 0, active: 1 },
    high_risk: { age: 62, gender: 2, height: 172, weight: 95, ap_hi: 165, ap_lo: 105, cholesterol: 3, gluc: 2, smoke: 1, alco: 1, active: 0 }
  };

  function applyPreset(key) {
    var p = PRESETS[key];
    if (!p) return;
    Object.assign(form, p);
    var setRange = function (id, v, unit) {
      var inp = document.getElementById("f_" + id);
      if (inp) { inp.value = v; }
      setTxt("v_" + id, v + " " + unit);
    };
    setRange("age", p.age, "years");
    setRange("height", p.height, "cm");
    setRange("weight", p.weight, "kg");
    setRange("hi", p.ap_hi, "mmHg");
    setRange("lo", p.ap_lo, "mmHg");

    var setSeg = function (id, val) {
      var host = document.getElementById(id);
      if (!host) return;
      Array.prototype.forEach.call(host.children, function (b) {
        b.setAttribute("aria-pressed", String(Number(b.dataset.v) === val));
      });
    };
    setSeg("f_gender", p.gender);
    setSeg("f_chol", p.cholesterol);
    setSeg("f_gluc", p.gluc);

    var lifeBtns = document.querySelectorAll("#f_life button");
    Array.prototype.forEach.call(lifeBtns, function (b) {
      var k = b.dataset.k;
      var on = Boolean(p[k]);
      b.setAttribute("aria-pressed", String(on));
    });
    predict();
  }

  document.addEventListener("click", function (e) {
    var btn = e.target.closest(".preset-btn");
    if (btn && btn.dataset.preset) {
      applyPreset(btn.dataset.preset);
    }
  });

  /* ---------------- boot ---------------- */
  (async function boot() {
    try {
      if (document.getElementById("dataStats")) { await loadSummary(); }
      if (document.getElementById("lossChart") || document.getElementById("coefChart")) { await loadModel(); }
      if (document.getElementById("rocChart")) { await loadRoc(); }
      if (document.getElementById("cm")) { await loadEvaluate(0.5); }
      if (document.getElementById("prob")) { await predict(); }
      if (document.getElementById("cmpTable")) { loadCompare().catch(function (e) { showApiError(e.message); }); }
    } catch (e) {
      showApiError("Could not reach the backend: " + e.message + ". Is the Flask server running?");
    }
  })();
})();
