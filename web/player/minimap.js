/* minimap.js — the whole-picture architecture map.
   Renders the trace as a grouped, click-to-jump list. Top-level
   groups (embed / block N / output); block groups reveal their
   sub-steps so you never lose your place in the ~120-op stack. */

function buildMinimap(host, steps, onJump) {
  host.innerHTML = '<div class="minimap-hint">map</div>';

  // group steps in order, preserving first-seen sequence
  const order = [];
  const byGroup = new Map();
  for (const s of steps) {
    const top = s.group.split("/")[0]; // embed | block-N | output
    if (!byGroup.has(top)) {
      byGroup.set(top, []);
      order.push(top);
    }
    byGroup.get(top).push(s);
  }

  const topLabel = (g) =>
    g === "embed" ? "embed" : g === "output" ? "output" : "block " + g.split("-")[1];

  for (const top of order) {
    const section = document.createElement("div");
    section.className = "mm-group";
    section.dataset.group = top;

    const head = document.createElement("div");
    head.className = "mm-head";
    head.textContent = topLabel(top);
    head.addEventListener("click", () => onJump(byGroup.get(top)[0].id));
    section.appendChild(head);

    const list = document.createElement("div");
    list.className = "mm-steps";
    for (const s of byGroup.get(top)) {
      const item = document.createElement("div");
      item.className = "mm-step";
      item.dataset.id = s.id;
      item.textContent = s.label;
      item.addEventListener("click", () => onJump(s.id));
      list.appendChild(item);
    }
    section.appendChild(list);
    host.appendChild(section);
  }
}

function setMinimapActive(host, stepId) {
  host.querySelectorAll(".mm-step").forEach((el) => {
    const on = Number(el.dataset.id) === stepId;
    el.classList.toggle("active", on);
    if (on) el.scrollIntoView({ block: "nearest" });
  });
}
