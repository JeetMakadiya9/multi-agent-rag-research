// Small DOM helpers. Every piece of server text is inserted as text, never as HTML.

export function append(node, ...children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : String(child));
  }
  return node;
}

export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "on") for (const [event, handler] of Object.entries(value)) node.addEventListener(event, handler);
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (typeof value === "boolean" && key in node) node[key] = value;
    else if (key === "value" && "value" in node) node.value = value;
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  return append(node, ...children);
}

export function clear(node) {
  node.replaceChildren();
  return node;
}

const SVG = "http://www.w3.org/2000/svg";
const CIRCLE = "M12 20.5a8.5 8.5 0 1 0 0-17 8.5 8.5 0 0 0 0 17Z";
const PATHS = {
  home: "M3.5 11.2 12 4l8.5 7.2M6 9.6v9.9h12V9.6",
  research: "M10.8 17.6a6.8 6.8 0 1 1 0-13.6 6.8 6.8 0 0 1 0 13.6ZM15.8 15.8 20.5 20.5",
  verify: "M12 3.5 19 6v5.6c0 4.1-2.9 7.6-7 8.9-4.1-1.3-7-4.8-7-8.9V6l7-2.5ZM8.9 12.1l2.1 2.1 4.2-4.4",
  library: "M5.5 4h3.2v16H5.5zM10.7 4h3.2v16h-3.2zM16 5.4l3.1-.8 3 14.9-3.1.8z",
  lab: "M9 3.5h6M10.2 3.5v6.2L5 18.3A1.5 1.5 0 0 0 6.3 20.5h11.4a1.5 1.5 0 0 0 1.3-2.2L13.8 9.7V3.5M7.6 14.5h8.8",
  sessions: `${CIRCLE}M12 7.6V12l2.9 1.8`,
  sun: "M12 15.8a3.8 3.8 0 1 0 0-7.6 3.8 3.8 0 0 0 0 7.6ZM12 2.8v2M12 19.2v2M5.5 5.5l1.4 1.4M17.1 17.1l1.4 1.4M2.8 12h2M19.2 12h2M5.5 18.5l1.4-1.4M17.1 6.9l1.4-1.4",
  moon: "M19.8 14.6A7.9 7.9 0 1 1 9.4 4.2a6.3 6.3 0 0 0 10.4 10.4Z",
  monitor: "M3.5 5h17v11h-17zM8.8 20h6.4M12 16v4",
  menu: "M4 6.5h16M4 12h16M4 17.5h16",
  close: "M6.5 6.5l11 11M17.5 6.5l-11 11",
  external: "M13.8 4.5h5.7v5.7M19.5 4.5l-8.4 8.4M17.2 13.6v4.6a1.3 1.3 0 0 1-1.3 1.3H5.8a1.3 1.3 0 0 1-1.3-1.3V8.1a1.3 1.3 0 0 1 1.3-1.3h4.6",
  upload: "M12 15.5V4.5M7.6 8.9 12 4.5l4.4 4.4M4.5 15v3.6a1.4 1.4 0 0 0 1.4 1.4h12.2a1.4 1.4 0 0 0 1.4-1.4V15",
  file: "M14 3.5H7.2a1.5 1.5 0 0 0-1.5 1.5v14a1.5 1.5 0 0 0 1.5 1.5h9.6a1.5 1.5 0 0 0 1.5-1.5V7.8L14 3.5ZM14 3.5v4.3h4.3",
  alert: "M12 4.2 20.8 19.5H3.2L12 4.2ZM12 10v4.3M12 16.9v.4",
  info: `${CIRCLE}M12 11v5M12 7.9v.4`,
  check: `${CIRCLE}M8.6 12.2l2.3 2.3 4.6-4.8`,
  minus: `${CIRCLE}M8.5 12h7`,
  cross: `${CIRCLE}M9.3 9.3l5.4 5.4M14.7 9.3l-5.4 5.4`,
  pending: `${CIRCLE}M12 12h.01`,
  panel: "M3.5 5h17v14h-17zM14.6 5v14",
  arrow: "M4.8 12h14.4M13.2 6l6 6-6 6",
  back: "M19.2 12H4.8M10.8 6l-6 6 6 6",
  spark: "M12 3.6l1.7 5 5 1.7-5 1.7-1.7 5-1.7-5-5-1.7 5-1.7Z",
  book: "M12 6.5C10.2 5.2 7.7 4.5 4.5 4.5v13c3.2 0 5.7.7 7.5 2 1.8-1.3 4.3-2 7.5-2v-13c-3.2 0-5.7.7-7.5 2ZM12 6.5v13",
  scale: "M12 4v16M7.5 20h9M5 7.5h14M5 7.5l-2.5 6a3 3 0 0 0 5 0L5 7.5ZM19 7.5l-2.5 6a3 3 0 0 0 5 0L19 7.5Z",
  refresh: "M19.5 12a7.5 7.5 0 1 1-2.2-5.3M19.5 4.5v4h-4",
  database: "M5 6.5C5 5.1 8.1 4 12 4s7 1.1 7 2.5S15.9 9 12 9 5 7.9 5 6.5ZM5 6.5v11c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5v-11M5 12c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5",
  compare: "M8.2 4.5v15M15.8 4.5v15M4.5 8.5h7.4M12.1 15.5h7.4",
  doc: "M6.5 3.5h11v17h-11zM9 8h6M9 11.5h6M9 15h4",
  pen: "M15.4 4.9l3.7 3.7L8.6 19.1H4.9v-3.7L15.4 4.9ZM13.3 7l3.7 3.7",
  quote: "M10 7.2c-2.8.8-4.5 3-4.5 6v3.6h4.4v-4.4H7.6M18.5 7.2c-2.8.8-4.5 3-4.5 6v3.6h4.4v-4.4h-2.3",
  copy: "M8.5 8.5h10.5v11H8.5zM15.5 8.5V5.6a1.1 1.1 0 0 0-1.1-1.1H6.1A1.1 1.1 0 0 0 5 5.6v9.3A1.1 1.1 0 0 0 6.1 16h2.4",
  stop: `${CIRCLE}M9.6 9.6h4.8v4.8H9.6z`,
  chevron: "M9.5 6l6 6-6 6",
  layers: "M12 4 3.5 8.4 12 12.8l8.5-4.4L12 4ZM3.5 12.4 12 16.8l8.5-4.4M3.5 16.2 12 20.6l8.5-4.4",
  list: "M9 6.5h10.5M9 12h10.5M9 17.5h10.5M4.8 6.5h.01M4.8 12h.01M4.8 17.5h.01",
  tick: "M6.2 12.6l3.7 3.7 7.9-8.6",
  x: "M7.6 7.6l8.8 8.8M16.4 7.6l-8.8 8.8",
  bang: "M12 6.2v7.4M12 17.3v.4",
  report: "M6.5 3.5h8l3 3v14h-11zM14.5 3.5v3h3M9 10.5h6M9 13.5h6M9 16.5h3.5",
  settings: "M4.5 7h8.5M17.5 7h2M4.5 12h2.5M11.5 12h8M4.5 17h10.5M19 17h.5M15 4.8v4.4M9 9.8v4.4M17 14.8v4.4",
  plug: "M9 3.5v4M15 3.5v4M6.5 7.5h11v3.2a5.5 5.5 0 0 1-11 0V7.5ZM12 16.2v4.3",
  user: "M12 12a3.8 3.8 0 1 0 0-7.6 3.8 3.8 0 0 0 0 7.6ZM4.8 20c.9-3.6 3.8-5.6 7.2-5.6s6.3 2 7.2 5.6",
  download: "M12 4.5v11M7.6 11.1 12 15.5l4.4-4.4M4.5 19.5h15",
  chevrondown: "M6.5 9.5l5.5 5.5 5.5-5.5",
  lock: "M6.5 10.5h11v9h-11zM8.8 10.5V8a3.2 3.2 0 0 1 6.4 0v2.5",
  chip: "M7.5 7.5h9v9h-9zM10 4v3.5M14 4v3.5M10 16.5V20M14 16.5V20M4 10h3.5M4 14h3.5M16.5 10H20M16.5 14H20",
  globe: `${CIRCLE}M3.5 12h17M12 3.5c2.3 2.4 3.4 5.2 3.4 8.5s-1.1 6.1-3.4 8.5c-2.3-2.4-3.4-5.2-3.4-8.5S9.7 5.9 12 3.5Z`,
  plus: "M12 5v14M5 12h14",
  cloud: "M7.5 18.5h9.8a3.7 3.7 0 0 0 .6-7.4 5.5 5.5 0 0 0-10.6 1.2A3.1 3.1 0 0 0 7.5 18.5Z",
};

export function icon(name, className = "") {
  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.6");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  if (className) svg.setAttribute("class", className);
  const path = document.createElementNS(SVG, "path");
  path.setAttribute("d", PATHS[name] || PATHS.info);
  svg.append(path);
  return svg;
}
