// Convert docs/report/REPORT_DRAFT.md (a restricted Markdown subset) to deliverables/report.docx.
// The output follows the Thai academic report layout: front cover, blank flyleaf, inner title page,
// preface, abstract, three tables of contents, chapters, bibliography, appendices.
// usage: node build_docx.js <repo root> <output.docx>
const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, Footer, Header, HeadingLevel, ImageRun, LevelFormat, NumberFormat,
  Packer, PageBreak, PageNumber, PageOrientation, Paragraph, ShadingType, StyleLevel, Table, TableCell,
  TableOfContents, TableRow, TextRun, WidthType,
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
const FIRST_LINE = 720;

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
    style: options.style,
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

// "รูปที่ 3.1 caption" and "ตารางที่ 3.1 caption": the label is bold, and the paragraph style is
// what the list of figures and the list of tables are built from.
function caption(label, text, style, alignment) {
  return new Paragraph({
    style, alignment, keepNext: style === "TableCaption",
    spacing: { before: style === "TableCaption" ? 160 : 60, after: style === "TableCaption" ? 80 : 200, line: 290 },
    children: [new TextRun({ text: `${label} `, bold: true, size: 20, color: INK, font: FONT }), ...runs(text, { size: 20 })],
  });
}

function image(file, text) {
  const { width, height } = pngSize(file);
  const scale = Math.min(620 / width, 560 / height, 1);
  const label = /^(รูปที่ \S+)\s+(.*)$/.exec(text);
  if (!label) throw new Error(`figure without a "รูปที่ N" label: ${text}`);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 60 },
      children: [new ImageRun({
        type: "png", data: fs.readFileSync(file),
        transformation: { width: Math.round(width * scale), height: Math.round(height * scale) },
        altText: { title: text, description: text, name: path.basename(file) },
      })],
    }),
    caption(label[1], label[2], "FigureCaption", AlignmentType.CENTER),
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

function table(rows, available, notesFollow = false) {
  const need = tableNeed(rows);
  const size = need > 110 ? 14 : need > 90 ? 15 : need > 70 ? 17 : 19;
  const weights = columnWeights(rows);
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  const widths = weights.map((weight) => Math.floor((weight / total) * available));
  widths[widths.length - 1] += available - widths.reduce((sum, width) => sum + width, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "BCCCDC" };
  const borders = { top: border, bottom: border, left: border, right: border };
  // A short table stays on one page with its caption; a long one may break between rows.
  const together = rows.length <= 14;
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
          keepNext: together && (notesFollow || rowIndex < rows.length - 1),
          children: runs(breakable(cell), { size, bold: rowIndex === 0 }),
        })],
      })),
    })),
  });
}

// A top-level heading starts a new page and is centred. "บทที่ 1 บทนำ" is set on two lines;
// Word joins them with a space in the table of contents.
function chapterHeading(text) {
  const chapter = /^(บทที่ \d+)\s+(.+)$/.exec(text);
  const children = chapter
    ? [new TextRun({ text: chapter[1], font: FONT }), new TextRun({ text: chapter[2], font: FONT, break: 1 })]
    : [new TextRun({ text, font: FONT })];
  return new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER, pageBreakBefore: true, children });
}

// ---- parse ---------------------------------------------------------------------------
const META_KEYS = {
  "ชื่อเรื่องภาษาอังกฤษ": "englishTitle", "ผู้จัดทำ": "authors", "เสนอ": "advisor",
  "รายวิชา": "course", "ภาคเรียน": "term", "สถาบัน": "institution",
};
const meta = { authors: [] };
const blocks = []; // { landscape: bool, kind, nodes: [...] }
function push(nodes, landscape = false, kind = "other") {
  blocks.push({ landscape, kind, nodes: Array.isArray(nodes) ? nodes : [nodes] });
}

let title = null;
let index = 0;
let inFront = true;
let generated = false;
let listInstance = 0;
let chapter = "";
let bodyStart = null; // index of the first block of chapter 1
let pendingTableCaption = null;

while (index < lines.length) {
  const line = lines[index];
  if (line.startsWith("<!-- BEGIN GENERATED")) { generated = true; index += 1; continue; }
  if (line.startsWith("<!-- END GENERATED")) { generated = false; index += 1; continue; }
  if (line.trim() === "") { index += 1; continue; }

  if (line.startsWith("# ")) { title = line.slice(2).trim(); index += 1; continue; }
  if (inFront && !line.startsWith("## ")) {
    const field = /^([^:]+):\s*(.+)$/.exec(line.trim());
    if (field && META_KEYS[field[1]]) {
      const key = META_KEYS[field[1]];
      if (key === "authors") meta.authors.push(field[2]); else meta[key] = field[2];
    }
    index += 1;
    continue;
  }
  inFront = false;

  const heading = /^(#{2,4}) (.+)$/.exec(line);
  // A generated table brings its own English title; the Thai caption above the block replaces it.
  if (heading && generated) { index += 1; continue; }
  if (heading) {
    const text = heading[2].trim();
    if (heading[1].length === 2) {
      chapter = text;
      if (bodyStart === null && /^บทที่ 1 /.test(text)) bodyStart = blocks.length;
      push(chapterHeading(text));
    } else {
      const level = heading[1].length === 3 ? HeadingLevel.HEADING_2 : HeadingLevel.HEADING_3;
      push(new Paragraph({ heading: level, children: [new TextRun({ text, font: FONT })] }));
    }
    index += 1;
    continue;
  }

  const picture = /^!\[(.*)\]\((.+)\)$/.exec(line.trim());
  if (picture) {
    push(image(path.resolve(path.dirname(source), picture[2]), picture[1]));
    index += 1;
    continue;
  }

  const tableCaption = /^\*\*(ตารางที่ [^*]+)\*\*\s*(.*)$/.exec(line.trim());
  if (tableCaption) {
    if (pendingTableCaption) throw new Error(`caption without a table: ${pendingTableCaption[1]}`);
    pendingTableCaption = tableCaption;
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
    if (!pendingTableCaption) throw new Error(`table without a "ตารางที่" caption, first row: ${rows[0].join(" | ")}`);
    const landscape = tableNeed(rows) > 132;
    const nodes = [caption(pendingTableCaption[1], pendingTableCaption[2], "TableCaption", AlignmentType.LEFT)];
    pendingTableCaption = null;
    // Notes that belong to a generated table stay with it, on the same page when the table is short.
    const notes = [];
    while (index < lines.length && (lines[index].trim() === "" || lines[index].startsWith(">"))) {
      if (lines[index].startsWith(">") && generated) {
        notes.push(lines[index].replace(/^>\s?/, ""));
      } else if (lines[index].startsWith(">")) {
        break;
      }
      index += 1;
    }
    nodes.push(table(rows, landscape ? LANDSCAPE_WIDTH : PORTRAIT_WIDTH, notes.length > 0));
    notes.forEach((note, position) => nodes.push(paragraph(note, {
      size: 17, italics: true, color: MUTED, after: 60, keepNext: position < notes.length - 1,
    })));
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
  const joined = text.join(" ");
  if (chapter === "บรรณานุกรม") {
    // One reference per paragraph, second line onwards indented.
    push(paragraph(joined, { indent: { left: FIRST_LINE, hanging: FIRST_LINE }, after: 140 }));
  } else if (chapter === "คำนำ" && joined.length < 30) {
    push(paragraph(joined, { alignment: AlignmentType.RIGHT, after: 60 }));
  } else if (joined.startsWith("ที่มา:")) {
    push(paragraph(joined, { size: 19, color: MUTED, after: 160 }), false, "paragraph");
  } else {
    push(paragraph(joined, { indent: { firstLine: FIRST_LINE } }), false, "paragraph");
  }
}
if (pendingTableCaption) throw new Error(`caption without a table: ${pendingTableCaption[1]}`);
if (bodyStart === null) throw new Error('no "## บทที่ 1 ..." heading found');

// ---- assemble ------------------------------------------------------------------------
const page = (landscape, pageNumbers) => ({
  page: {
    size: { width: 11906, height: 16838, orientation: landscape ? PageOrientation.LANDSCAPE : PageOrientation.PORTRAIT },
    margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 },
    pageNumbers,
  },
});
const header = new Header({ children: [new Paragraph({
  alignment: AlignmentType.RIGHT,
  children: [new TextRun({ text: "QQQ Volatility Forecasting", font: FONT, size: 16, color: MUTED })],
})] });
const pageNumberFooter = () => new Footer({ children: [new Paragraph({
  alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: MUTED })],
})] });

const centred = (text, size, { bold = false, before = 0, after = 160 } = {}) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before, after, line: 340 },
  children: [new TextRun({ text, font: FONT, size, bold, color: INK })],
});

// The front cover and the inner title page carry the same information.
function titlePage() {
  return [
    centred("รายงานโครงงาน", 30, { bold: true, before: 1500, after: 500 }),
    centred(title, 32, { bold: true, after: 200 }),
    centred(meta.englishTitle || "", 24, { after: 1500 }),
    centred("จัดทำโดย", 24, { after: 200 }),
    ...meta.authors.map((author) => centred(author, 24, { after: 120 })),
    centred("เสนอ", 24, { before: 1100, after: 200 }),
    centred(meta.advisor || "", 24, { after: 1500 }),
    centred(meta.course || "", 22, { after: 120 }),
    centred(meta.term || "", 22, { after: 120 }),
    centred(meta.institution || "", 22, { after: 0 }),
  ];
}
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });
const coverSection = [...titlePage(), pageBreak(), pageBreak(), ...titlePage()];

// "สารบัญ" itself is not a heading, so it does not list itself.
const contents = [
  new Paragraph({ style: "FrontTitle", alignment: AlignmentType.CENTER, pageBreakBefore: true,
    children: [new TextRun({ text: "สารบัญ", font: FONT })] }),
  new TableOfContents("สารบัญ", { hyperlink: true, headingStyleRange: "1-2" }),
  chapterHeading("สารบัญรูป"),
  new TableOfContents("สารบัญรูป", { hyperlink: true, stylesWithLevels: [new StyleLevel("FigureCaption", 1)] }),
  chapterHeading("สารบัญตาราง"),
  new TableOfContents("สารบัญตาราง", { hyperlink: true, stylesWithLevels: [new StyleLevel("TableCaption", 1)] }),
];

blocks.forEach((block, position) => {
  const before = blocks[position - 1];
  const after = blocks[position + 1];
  if (block.kind === "paragraph" && before && after && before.landscape && after.landscape) block.landscape = true;
});

const frontBlocks = blocks.slice(0, bodyStart);
if (frontBlocks.some((block) => block.landscape)) throw new Error("the preface and abstract cannot hold a wide table");
const frontSection = [...frontBlocks.flatMap((block) => block.nodes), ...contents];

const bodySections = [];
let current = null;
for (const block of blocks.slice(bodyStart)) {
  if (!current || block.landscape !== current.landscape) {
    current = { landscape: block.landscape, children: [] };
    bodySections.push(current);
  }
  current.children.push(...block.nodes);
}

const headingStyle = (id, name, size, before, after) => ({
  id, name, basedOn: "Normal", next: "Normal", quickFormat: true,
  run: { size, bold: true, font: FONT, color: id === "Heading1" ? INK : ACCENT },
  paragraph: { spacing: { before, after, line: 340 }, keepNext: true, outlineLevel: Number(id.slice(-1)) - 1 },
});
const plainStyle = (id, run, paragraphOptions = {}) => ({
  id, name: id, basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, ...run }, paragraph: paragraphOptions,
});

const document = new Document({
  creator: "QQQ Volatility Forecasting team",
  title,
  description: "Project report generated from docs/report/REPORT_DRAFT.md",
  styles: {
    default: { document: { run: { font: FONT, size: BODY, color: INK } } },
    paragraphStyles: [
      headingStyle("Heading1", "Heading 1", 34, 0, 360),
      headingStyle("Heading2", "Heading 2", 26, 300, 140),
      headingStyle("Heading3", "Heading 3", 23, 220, 120),
      plainStyle("FrontTitle", { size: 34, bold: true, color: INK }, { spacing: { after: 360, line: 340 } }),
      plainStyle("FigureCaption", { size: 20, color: INK }),
      plainStyle("TableCaption", { size: 20, color: INK }),
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

  sections: [
    // Covers and flyleaf: no header, no page number.
    { properties: page(false), children: coverSection },
    // Preface to the list of tables: Thai-letter page numbers (ก, ข, ค, ...).
    { properties: page(false, { start: 1, formatType: NumberFormat.THAI_LETTERS }),
      footers: { default: pageNumberFooter() }, children: frontSection },
    // Chapters onwards: arabic numbers from 1, continuing across portrait and landscape sections.
    ...bodySections.map((section, position) => ({
      properties: page(section.landscape, position === 0
        ? { start: 1, formatType: NumberFormat.DECIMAL } : { formatType: NumberFormat.DECIMAL }),
      headers: { default: header },
      footers: { default: pageNumberFooter() },
      children: section.children,
    })),
  ],
});

Packer.toBuffer(document).then((buffer) => {
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, buffer);
  console.log(`wrote ${output} (${buffer.length} bytes, ${bodySections.length + 2} sections)`);
});
