// Product identity: the mark and the wordmark, kept separate so each can be used on its own.
// The mark is a source page with one line drawn out of it onto the next page: evidence carried from a source into
// your notes. Colours come from the theme tokens, so it follows light and dark mode; favicon.svg is the same mark.
import { el } from "./dom.mjs";

export const PRODUCT = { name: "Research Studio", full: "Multi-Agent Research Studio", tagline: "Evidence-grounded research, on your computer" };

const SVG = "http://www.w3.org/2000/svg";
let counter = 0;

function node(tag, attrs, ...children) {
  const n = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  n.append(...children);
  return n;
}

export function logoMark({ size = 32, label = "" } = {}) {
  counter += 1;
  const cut = `logo-cut-${counter}`;
  const behind = `logo-behind-${counter}`;
  const svg = node("svg", { viewBox: "0 0 32 32", width: size, height: size, class: "logo-mark", focusable: "false",
    ...(label ? { role: "img", "aria-label": label } : { "aria-hidden": "true" }) },
  node("defs", {},
    node("mask", { id: cut },
      node("rect", { width: 32, height: 32, fill: "#fff" }),
      node("path", { d: "M8.6 14h7.6M8.6 19.2h13.4M8.6 24.4h4.6", stroke: "#000", "stroke-width": 2.3, "stroke-linecap": "round" })),
    node("mask", { id: behind },
      node("rect", { width: 32, height: 32, fill: "#fff" }),
      node("rect", { x: 2.4, y: 6.4, width: 20.2, height: 24.2, rx: 4.8, fill: "#000" }))),
  node("rect", { class: "logo-back", x: 11, y: 2.6, width: 17.4, height: 21.4, rx: 3.6, "stroke-width": 2, mask: `url(#${behind})` }),
  node("rect", { class: "logo-sheet", x: 4, y: 8, width: 17, height: 21, rx: 3.4, mask: `url(#${cut})` }),
  node("path", { class: "logo-trail", d: "M22.4 19.2h0.4", "stroke-width": 2.3, "stroke-linecap": "round" }),
  node("circle", { class: "logo-dot", cx: 25.2, cy: 19.2, r: 2.7 }));
  return svg;
}

export function wordmark({ sub = true } = {}) {
  return el("span", { class: "wordmark" }, el("span", { class: "wordmark-name" }, PRODUCT.name),
    sub ? el("span", { class: "wordmark-sub" }, "Multi-agent RAG") : null);
}

export function brandLink(href = "#/") {
  return el("a", { class: "brand", href, "aria-label": `${PRODUCT.name}, home` }, logoMark({ size: 34 }), wordmark());
}
