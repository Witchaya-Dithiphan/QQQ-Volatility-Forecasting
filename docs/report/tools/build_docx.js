// Convert docs/report/REPORT_DRAFT.md (a restricted Markdown subset) to deliverables/report.docx.
// usage: node build_docx.js <repo root> <output.docx>
const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, Footer, Header, HeadingLevel, ImageRun, LevelFormat, Packer,
  PageBreak, PageNumber, PageOrientation, Paragraph, ShadingType, Table, TableCell, TableOfContents,
  TableRow, TextRun, WidthType,
} = require("docx");

const root = process.argv[2];
const output = process.argv[3];
const source = path.join(root, "docs", "report", "REPORT_DRAFT.md");
const lines = fs.readFileSync(source, "utf8").replace(/\r\n/g, "\n").split("\n");

const FONT = "Leelawadee UI";
const MONO = "Consolas";
const BODY = 21; // half-points
const INK = "1F2933";
const MUTED = "52606D";
const ACCENT = "1F4E79";
const PORTRAIT_WIDTH = 11906 - 2 * 1134;
const LANDSCAPE_WIDTH = 16838 - 2 * 1134;

function runs(text, { size = BODY, bold = false, italics = false, color = INK } = {}) {
  // **bold** and `code` only; everything else is literal.
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter((part) => part !== "");
  return parts.map((part) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return new TextRun({ text: part.slice(2, -2), bold: true, italics, size, color, font: FONT });
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return new TextRun({ text: part.slice(1, -1), bold, italics, size: size - 1, color: "243B53", font: MONO });
    }
    return new TextRun({ text: part, bold, italics, size, color, font: FONT });
  });
}

function paragraph(text, options = {}) {
  return new Paragraph({
    children: runs(text, options),
    spacing: { after: options.after ?? 120, line: 300 },
    alignment: options.alignment,
    keepNext: options.keepNext,
    indent: options.indent,
    border: options.border,
    shading: options.shading,
  });
}

function pngSize(file) {
  const buffer = fs.readFileSync(file);
  return { width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20) };
}

function image(file, caption) {
  const { width, height } = pngSize(file);
  const scale = Math.min(620 / width, 560 / height, 1);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 60 },
      children: [new ImageRun({
        type: "png", data: fs.readFileSync(file),
        transformation: { width: Math.round(width * scale), height: Math.round(height * scale) },
        altText: { title: caption, description: caption, name: path.basename(file) },
      })],
    }),
    paragraph(caption, { size: 19, italics: true, color: MUTED, alignment: AlignmentType.CENTER, after: 200 }),
  ];
}

function splitRow(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
}

// Cells may wrap after "_" and "," (a zero-width space is inserted there), so a column
// only has to be as wide as its longest unbreakable piece.
const ZWSP = "​";
const breakable = (text) => text.replace(/([_,/])(?=\S)/g, `$1${ZWSP}`).replace(/\[/g, `${ZWSP}[`);

function columnWeights(rows) {
  return rows[0].map((_, index) => {
    const cells = rows.map((row) => row[index] || "");
    const piece = Math.max(...cells.map((cell) => Math.max(...cell.replace(/`/g, "").split(/[\s_,/[]+/).map((word) => word.length))));
    const data = Math.max(...cells.slice(1).map((cell) => cell.replace(/`/g, "").length), 0);
    return Math.max(piece, Math.min(data, data > 60 ? 40 : 12), 4);
  });
}

const tableNeed = (rows) => columnWeights(rows).reduce((sum, weight) => sum + weight, 0) + 1.5 * rows[0].length;

function table(rows, available) {
  const need = tableNeed(rows);
  const size = need > 110 ? 14 : need > 90 ? 15 : need > 70 ? 17 : 19;
  const weights = columnWeights(rows);
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  const widths = weights.map((weight) => Math.floor((weight / total) * available));
  widths[widths.length - 1] += available - widths.reduce((sum, width) => sum + width, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "BCCCDC" };
  const borders = { top: border, bottom: border, left: border, right: border };
  return new Table({
    width: { size: available, type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map((row, rowIndex) => new TableRow({
      tableHeader: rowIndex === 0,
      cantSplit: true,
      children: row.map((cell, index) => new TableCell({
        width: { size: widths[index], type: WidthType.DXA },
        borders,
        margins: { top: 50, bottom: 50, left: 70, right: 70 },
        shading: rowIndex === 0 ? { type: ShadingType.CLEAR, fill: "E4ECF5", color: "auto" } : undefined,
        children: [new Paragraph({
          spacing: { after: 0, line: 250 },
          children: runs(breakable(cell), { size, bold: rowIndex === 0 }),
        })],
      })),
    })),
  });
}

// ---- parse ---------------------------------------------------------------------------
const blocks = []; // { landscape: bool, nodes: [...] }
function push(nodes, landscape = false, kind = "other") {
  blocks.push({ landscape, kind, nodes: Array.isArray(nodes) ? nodes : [nodes] });
}

let title = null;
const front = [];
let index = 0;
let inFront = true;
let generated = false;
let listInstance = 0;
let pendingCaption = null;

while (index < lines.length) {
  const line = lines[index];
  if (line.startsWith("<!-- BEGIN GENERATED")) { generated = true; index += 1; continue; }
  if (line.startsWith("<!-- END GENERATED")) { generated = false; index += 1; continue; }
  if (line.trim() === "") { index += 1; continue; }

  if (line.startsWith("# ")) { title = line.slice(2).trim(); index += 1; continue; }
  if (inFront && !line.startsWith("## ")) {
    if (!line.startsWith(">")) front.push(line.trim());
    index += 1;
    continue;
  }
  inFront = false;

  const heading = /^(#{2,4}) (.+)$/.exec(line);
  if (heading && generated) { pendingCaption = heading[2].trim(); index += 1; continue; }
  if (heading) {
    const level = { 2: HeadingLevel.HEADING_1, 3: HeadingLevel.HEADING_2, 4: HeadingLevel.HEADING_3 }[heading[1].length];
    push(new Paragraph({ heading: level, children: [new TextRun({ text: heading[2].trim(), font: FONT })] }));
    index += 1;
    continue;
  }

  const picture = /^!\[(.*)\]\((.+)\)$/.exec(line.trim());
  if (picture) {
    push(image(path.resolve(path.dirname(source), picture[2]), picture[1]));
    index += 1;
    continue;
  }

  if (line.trim().startsWith("|")) {
    const rows = [];
    while (index < lines.length && lines[index].trim().startsWith("|")) {
      const cells = splitRow(lines[index]);
      if (!cells.every((cell) => /^:?-{3,}:?$/.test(cell))) rows.push(cells);
      index += 1;
    }
    const landscape = tableNeed(rows) > 132;
    const nodes = [];
    if (pendingCaption) {
      nodes.push(paragraph(pendingCaption, { size: 19, bold: true, color: ACCENT, keepNext: true, after: 80 }));
      pendingCaption = null;
    }
    nodes.push(table(rows, landscape ? LANDSCAPE_WIDTH : PORTRAIT_WIDTH));
    // Notes that belong to a generated table stay with it.
    while (index < lines.length && (lines[index].trim() === "" || lines[index].startsWith(">"))) {
      if (lines[index].startsWith(">") && generated) {
        nodes.push(paragraph(lines[index].replace(/^>\s?/, ""), { size: 17, italics: true, color: MUTED, after: 60 }));
      } else if (lines[index].startsWith(">")) {
        break;
      }
      index += 1;
    }
    nodes.push(new Paragraph({ spacing: { after: 160 }, children: [] }));
    push(nodes, landscape);
    continue;
  }

  if (line.startsWith(">")) {
    const text = [];
    while (index < lines.length && lines[index].startsWith(">")) { text.push(lines[index].replace(/^>\s?/, "")); index += 1; }
    push(paragraph(text.join(" "), {
      indent: { left: 300, right: 200 }, after: 180,
      border: { left: { style: BorderStyle.SINGLE, size: 18, color: ACCENT, space: 10 } },
      shading: { type: ShadingType.CLEAR, fill: "F0F4F8", color: "auto" },
    }));
    continue;
  }

  const bullet = /^(\s*)- (.+)$/.exec(line);
  const numbered = /^(\s*)(\d+)\. (.+)$/.exec(line);
  if (bullet || numbered) {
    const ordered = Boolean(numbered);
    listInstance += 1;
    const nodes = [];
    while (index < lines.length) {
      const item = ordered ? /^(\s*)(\d+)\. (.+)$/.exec(lines[index]) : /^(\s*)- (.+)$/.exec(lines[index]);
      if (!item) break;
      const text = ordered ? item[3] : item[2];
      nodes.push(new Paragraph({
        children: runs(text),
        numbering: { reference: ordered ? "numbers" : "bullets", level: item[1].length >= 2 ? 1 : 0, instance: listInstance },
        spacing: { after: 70, line: 300 },
      }));
      index += 1;
    }
    nodes.push(new Paragraph({ spacing: { after: 60 }, children: [] }));
    push(nodes);
    continue;
  }

  // A line that is only inline code is a formula: centre it.
  if (/^`[^`]+`$/.test(line.trim())) {
    push(new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 80, after: 160 },
      shading: { type: ShadingType.CLEAR, fill: "F0F4F8", color: "auto" },
      children: [new TextRun({ text: line.trim().slice(1, -1), font: MONO, size: BODY, color: "102A43" })],
    }));
    index += 1;
    continue;
  }

  const text = [line.trim()];
  index += 1;
  while (index < lines.length && lines[index].trim() !== "" && !/^(#|\||>|!\[|\s*- |\s*\d+\. |<!--)/.test(lines[index])) {
    text.push(lines[index].trim());
    index += 1;
  }
  push(paragraph(text.join(" ")), false, "paragraph");
}

// ---- assemble ------------------------------------------------------------------------
const page = (landscape) => ({
  page: {
    size: { width: 11906, height: 16838, orientation: landscape ? PageOrientation.LANDSCAPE : PageOrientation.PORTRAIT },
    margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 },
  },
});
const header = new Header({ children: [new Paragraph({
  alignment: AlignmentType.RIGHT,
  children: [new TextRun({ text: "QQQ Volatility Forecasting", font: FONT, size: 16, color: MUTED })],
})] });
const footer = new Footer({ children: [new Paragraph({
  alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: MUTED })],
})] });

const cover = [
  new Paragraph({ spacing: { before: 3200 }, children: [] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 },
    children: [new TextRun({ text: title, font: FONT, size: 44, bold: true, color: ACCENT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 900 },
    children: [new TextRun({ text: "QQQ Volatility Forecasting", font: FONT, size: 28, color: MUTED })] }),
  ...front.map((text) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
    children: runs(text, { size: 24 }) })),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: "สารบัญ", font: FONT })] }),
  new TableOfContents("สารบัญ", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new PageBreak()] }),
];

blocks.forEach((block, position) => {
  const before = blocks[position - 1];
  const after = blocks[position + 1];
  if (block.kind === "paragraph" && before && after && before.landscape && after.landscape) block.landscape = true;
});

const sections = [];
let current = { landscape: false, children: [...cover] };
for (const block of blocks) {
  if (block.landscape !== current.landscape) {
    sections.push(current);
    current = { landscape: block.landscape, children: [] };
  }
  current.children.push(...block.nodes);
}
sections.push(current);

const headingStyle = (id, name, size, before, color) => ({
  id, name, basedOn: "Normal", next: "Normal", quickFormat: true,
  run: { size, bold: true, font: FONT, color },
  paragraph: { spacing: { before, after: 140 }, keepNext: true, outlineLevel: Number(id.slice(-1)) - 1 },
});

const document = new Document({
  creator: "QQQ Volatility Forecasting team",
  title,
  description: "Project report generated from docs/report/REPORT_DRAFT.md",
  styles: {
    default: { document: { run: { font: FONT, size: BODY, color: INK } } },
    paragraphStyles: [
      headingStyle("Heading1", "Heading 1", 32, 420, ACCENT),
      headingStyle("Heading2", "Heading 2", 26, 300, ACCENT),
      headingStyle("Heading3", "Heading 3", 23, 220, INK),
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 1000, hanging: 270 } } } }] },
      { reference: "numbers", levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 540, hanging: 360 } } } },
        { level: 1, format: LevelFormat.LOWER_LETTER, text: "%2.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 1000, hanging: 360 } } } }] },
    ],
  },

  sections: sections.map((section) => ({
    properties: page(section.landscape),
    headers: { default: header },
    footers: { default: footer },
    children: section.children,
  })),
});

Packer.toBuffer(document).then((buffer) => {
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, buffer);
  console.log(`wrote ${output} (${buffer.length} bytes, ${sections.length} sections)`);
});
