// Convert docs/report/REPORT_DRAFT.md (a restricted Markdown subset) to deliverables/report.docx.
// The output follows the Thai academic report layout: front cover, preface, abstract, three
// tables of contents, chapters, bibliography, appendices. Everything is set in TH SarabunPSK,
// body text 16 pt, headings larger and bold, all in black with no shading or coloured text.
// usage: node build_docx.js <repo root> <output.docx>
const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, Footer, HeadingLevel, ImageRun, LevelFormat, NumberFormat,
  Packer, PageNumber, PageOrientation, Paragraph, StyleLevel, Table, TableCell,
  TableOfContents, TableRow, TextRun, WidthType,
} = require("docx");

const root = process.argv[2];
const output = process.argv[3];
const source = path.join(root, "docs", "report", "REPORT_DRAFT.md");
const lines = fs.readFileSync(source, "utf8").replace(/\r\n/g, "\n").split("\n");

const FONT = "TH SarabunPSK";
const BODY = 32; // half-points: 16 pt
const SMALL = 28; // 14 pt, for source lines and table notes
const INK = "000000";
// Thai academic margins: 1.5 inch at the top and binding side, 1 inch elsewhere. A landscape
// page holds only a wide table, so it keeps 1 inch all round to give the table more room.
const MARGIN = { top: 2160, bottom: 1440, left: 2160, right: 1440 };
const LANDSCAPE_MARGIN = { top: 1440, bottom: 1440, left: 1440, right: 1440 };
const PORTRAIT_WIDTH = 11906 - MARGIN.left - MARGIN.right;
const LANDSCAPE_WIDTH = 16838 - LANDSCAPE_MARGIN.left - LANDSCAPE_MARGIN.right;
const FIRST_LINE = 720;
const LINE = 240; // single spacing: TH SarabunPSK already carries generous leading
// Thai text is tagged as Thai so Word can apply its own rules where they are installed.
const LANGUAGE = { value: "en-US", bidirectional: "th-TH" };

// Thai is written without spaces between words. A Word without the Thai editing language can
// only break a line at a space, which leaves ragged lines, so a zero-width space is put at every
// Thai word boundary that ICU's dictionary finds. The text itself is unchanged.
const ZWSP = "​";
const THAI = /[฀-๿]/;
const segmenter = new Intl.Segmenter("th", { granularity: "word" });
function thaiBreaks(text) {
  let out = "";
  let previous = "";
  for (const { segment } of segmenter.segment(text)) {
    if (previous && THAI.test(previous.slice(-1)) && THAI.test(segment[0])) out += ZWSP;
    out += segment;
    previous = segment;
  }
  return out;
}

// TH SarabunPSK has no Greek letters and few math signs. Word would substitute a font of its
// own choosing at the full 16 pt, which looks bold and oversized next to Sarabun, so these
// characters are set in Cambria (or Segoe UI Symbol) at three quarters of the size around them.
const SYMBOLS = /([ŷͰ-Ͽᵀ‖⁰-₟←-⇿⊙★]+)/;
const symbolFont = (text) => (/[⊙★]/.test(text) ? "Segoe UI Symbol" : "Cambria");
function textRuns(text, { size = BODY, ...options }) {
  return text.split(SYMBOLS).filter((part) => part !== "").map((part) => (SYMBOLS.test(part)
    ? new TextRun({ language: LANGUAGE, text: part, ...options, size: Math.round(size * 0.75), font: symbolFont(part) })
    : new TextRun({ language: LANGUAGE, text: part, ...options, size, font: FONT })));
}

function runs(text, { size = BODY, bold = false, italics = false, color = INK } = {}) {
  // Print link labels without Markdown URLs; preserve inline formatting inside labels.
  const parts = text.split(/(\[[^\]]+\]\([^)]+\)|\*\*[^*]+\*\*|`[^`]+`)/g).filter((part) => part !== "");
  return parts.flatMap((part) => {
    if (part.startsWith("[") && part.includes("](")) {
      return runs(part.slice(1, part.indexOf("](")), { size, bold, italics, color });
    }
    if (part.startsWith("**") && part.endsWith("**")) {
      return textRuns(thaiBreaks(part.slice(2, -2)), { bold: true, italics, size, color });
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      // A long identifier may wrap after "_" or "/" instead of stretching the line around it.
      return textRuns(part.slice(1, -1).replace(/([_/])(?=\S)/g, `$1${ZWSP}`), { bold, italics, size, color });
    }
    return textRuns(thaiBreaks(part), { bold, italics, size, color });
  });
}

function paragraph(text, options = {}) {
  return new Paragraph({
    children: runs(text, options),
    style: options.style,
    spacing: { after: options.after ?? 120, line: LINE },
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
    spacing: { before: style === "TableCaption" ? 160 : 60, after: style === "TableCaption" ? 80 : 200, line: LINE },
    children: [new TextRun({ language: LANGUAGE, text: `${label} `, bold: true, size: BODY, color: INK, font: FONT }), ...runs(text, { size: BODY })],
  });
}

function image(file, text) {
  const { width, height } = pngSize(file);
  // 550 px is the text width between the margins at 96 dpi; the height leaves room for the caption.
  const scale = Math.min(550 / width, 640 / height, 1);
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

function table(rows, available, notesFollow = false, landscape = false) {
  const need = tableNeed(rows);
  const size = need > 110 ? 22 : need > 90 ? 24 : need > 70 ? 26 : 28;
  const weights = columnWeights(rows);
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  const widths = weights.map((weight) => Math.floor((weight / total) * available));
  widths[widths.length - 1] += available - widths.reduce((sum, width) => sum + width, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: INK };
  const borders = { top: border, bottom: border, left: border, right: border };
  // A short table stays on one page with its caption. A long or wide one may break between rows,
  // but the caption keeps the first rows with it and the last three rows stay together.
  const together = !landscape && rows.length <= 14;
  const keep = (rowIndex) => (together
    ? notesFollow || rowIndex < rows.length - 1
    : rowIndex < 2 || (rowIndex >= rows.length - 3 && (notesFollow || rowIndex < rows.length - 1)));
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
        children: [new Paragraph({
          spacing: { after: 0, line: LINE },
          keepNext: keep(rowIndex),
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
    ? [new TextRun({ language: LANGUAGE, text: thaiBreaks(chapter[1]), font: FONT }), new TextRun({ language: LANGUAGE, text: thaiBreaks(chapter[2]), font: FONT, break: 1 })]
    : [new TextRun({ language: LANGUAGE, text: thaiBreaks(text), font: FONT })];
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
      push(chapterHeading(text), false, "chapter");
    } else {
      const level = heading[1].length === 3 ? HeadingLevel.HEADING_2 : HeadingLevel.HEADING_3;
      push(new Paragraph({ heading: level, children: textRuns(thaiBreaks(text), { size: heading[1].length === 3 ? 36 : 34 }) }), false, "heading");
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
    nodes.push(table(rows, landscape ? LANDSCAPE_WIDTH : PORTRAIT_WIDTH, notes.length > 0, landscape));
    notes.forEach((note, position) => nodes.push(paragraph(note, {
      size: SMALL, italics: true, after: 60, keepNext: position < notes.length - 1,
    })));
    nodes.push(new Paragraph({ spacing: { after: 160 }, children: [] }));
    push(nodes, landscape);
    continue;
  }

  if (line.startsWith(">")) {
    const text = [];
    while (index < lines.length && lines[index].startsWith(">")) { text.push(lines[index].replace(/^>\s?/, "")); index += 1; }
    push(paragraph(text.join(" "), { indent: { left: 720, right: 360 }, after: 180 }));
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
      // Indented bullets under a numbered item stay in the same list, so the numbering carries on after them.
      const nested = ordered && !item ? /^(\s{2,})- (.+)$/.exec(lines[index]) : null;
      if (!item && !nested) break;
      nodes.push(new Paragraph({
        children: runs(nested ? nested[2] : ordered ? item[3] : item[2]),
        numbering: nested
          ? { reference: "bullets", level: 1, instance: listInstance }
          : { reference: ordered ? "numbers" : "bullets", level: item[1].length >= 2 ? 1 : 0, instance: listInstance },
        spacing: { after: 70, line: LINE },
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
      alignment: AlignmentType.CENTER, spacing: { before: 80, after: 160, line: LINE },
      children: textRuns(line.trim().slice(1, -1), { size: BODY, color: INK }),
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
    push(paragraph(joined, { size: SMALL, after: 160 }), false, "paragraph");
  } else {
    // Body text is justified the Thai way, which spreads the spacing within each line.
    push(paragraph(joined, { indent: { firstLine: FIRST_LINE }, alignment: AlignmentType.THAI_DISTRIBUTE }), false, "paragraph");
  }
}
if (pendingTableCaption) throw new Error(`caption without a table: ${pendingTableCaption[1]}`);
if (bodyStart === null) throw new Error('no "## บทที่ 1 ..." heading found');

// ---- assemble ------------------------------------------------------------------------
const page = (landscape, pageNumbers) => ({
  page: {
    size: { width: 11906, height: 16838, orientation: landscape ? PageOrientation.LANDSCAPE : PageOrientation.PORTRAIT },
    margin: landscape ? LANDSCAPE_MARGIN : MARGIN,
    pageNumbers,
  },
});
const pageNumberFooter = () => new Footer({ children: [new Paragraph({
  alignment: AlignmentType.CENTER,
  children: [new TextRun({ language: LANGUAGE, children: [PageNumber.CURRENT], font: FONT, size: BODY, color: INK })],
})] });

const centred = (text, size, { bold = false, before = 0, after = 160 } = {}) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before, after, line: LINE },
  children: [new TextRun({ language: LANGUAGE, text, font: FONT, size, bold, color: INK })],
});

// "นายพิทวัส เฉิน 67011509" becomes three aligned columns: first name, surname, student ID.
function authorTable() {
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const borders = { top: none, bottom: none, left: none, right: none };
  const widths = [1900, 1900, 1500];
  const cell = (text, width) => new TableCell({
    width: { size: width, type: WidthType.DXA }, borders,
    children: [new Paragraph({ spacing: { after: 80, line: LINE },
      children: [new TextRun({ language: LANGUAGE, text, font: FONT, size: BODY, color: INK })] })],
  });
  return new Table({
    alignment: AlignmentType.CENTER, width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    borders: { top: none, bottom: none, left: none, right: none, insideHorizontal: none, insideVertical: none },
    rows: meta.authors.map((author) => {
      const parts = author.trim().split(/\s+/);
      const id = /^\d+$/.test(parts[parts.length - 1]) ? parts.pop() : "";
      return new TableRow({ children: [cell(parts[0] || "", widths[0]), cell(parts.slice(1).join(" "), widths[1]), cell(id, widths[2])] });
    }),
  });
}

// One front cover; the blank flyleaf and the inner title page were removed by the authors.
function titlePage() {
  return [
    centred("รายงานโครงงาน", 40, { bold: true, before: 600, after: 600 }),
    centred(title, 40, { bold: true, after: 200 }),
    centred(meta.englishTitle || "", BODY, { after: 1400 }),
    centred("จัดทำโดย", BODY, { bold: true, after: 200 }),
    authorTable(),
    centred("เสนอ", BODY, { bold: true, before: 1200, after: 200 }),
    centred(meta.advisor || "", BODY, { after: 1400 }),
    centred(meta.course || "", BODY, { after: 120 }),
    centred(meta.term || "", BODY, { after: 120 }),
    centred(meta.institution || "", BODY, { after: 0 }),
  ];
}
const coverSection = titlePage();

// "สารบัญ" itself is not a heading, so it does not list itself.
const contents = [
  new Paragraph({ style: "FrontTitle", alignment: AlignmentType.CENTER, pageBreakBefore: true,
    children: [new TextRun({ language: LANGUAGE, text: "สารบัญ", font: FONT })] }),
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
// A heading (and its opening paragraph) right before a wide table moves onto the table's
// landscape page, so the portrait page before it is not left holding a heading alone.
blocks.forEach((block, position) => {
  if (!block.landscape || (blocks[position - 1] && blocks[position - 1].landscape)) return;
  let start = position;
  while (start > 0 && position - start < 3 && ["heading", "chapter", "paragraph"].includes(blocks[start - 1].kind)) {
    start -= 1;
    if (blocks[start].kind === "chapter") break;
  }
  while (start < position && blocks[start].kind === "paragraph") start += 1;
  for (let moved = start; moved < position; moved += 1) blocks[moved].landscape = true;
});

const frontBlocks = blocks.slice(0, bodyStart);
if (frontBlocks.some((block) => block.landscape)) throw new Error("the preface and abstract cannot hold a wide table");
const frontSection = [...frontBlocks.flatMap((block) => block.nodes), ...contents];

const bodySections = [];
let current = null;
for (const block of blocks.slice(bodyStart)) {
  // A chapter that opens on a landscape page after another landscape page starts its own section:
  // inside one section Word adds a blank page before the chapter's page break.
  if (!current || block.landscape !== current.landscape || (block.landscape && block.kind === "chapter")) {
    current = { landscape: block.landscape, children: [] };
    bodySections.push(current);
  }
  current.children.push(...block.nodes);
}

const headingStyle = (id, name, size, before, after) => ({
  id, name, basedOn: "Normal", next: "Normal", quickFormat: true,
  run: { size, bold: true, font: FONT, color: INK },
  paragraph: { spacing: { before, after, line: LINE }, keepNext: true, outlineLevel: Number(id.slice(-1)) - 1 },
});
const plainStyle = (id, run, paragraphOptions = {}) => ({
  id, name: id, basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, ...run }, paragraph: paragraphOptions,
});

const document = new Document({
  creator: "QQQ Volatility Forecasting team",
  title,
  description: "Project report generated from docs/report/REPORT_DRAFT.md",
  styles: {
    default: { document: { run: { font: FONT, size: BODY, color: INK, language: LANGUAGE } } },
    paragraphStyles: [
      headingStyle("Heading1", "Heading 1", 40, 0, 360),
      headingStyle("Heading2", "Heading 2", 36, 280, 140),
      headingStyle("Heading3", "Heading 3", 34, 200, 100),
      plainStyle("FrontTitle", { size: 40, bold: true, color: INK }, { spacing: { after: 360, line: LINE } }),
      plainStyle("FigureCaption", { size: BODY, color: INK }),
      plainStyle("TableCaption", { size: BODY, color: INK }),
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
    // Front cover: no page number.
    { properties: page(false), children: coverSection },
    // Preface to the list of tables: Thai-letter page numbers (ก, ข, ค, ...).
    { properties: page(false, { start: 1, formatType: NumberFormat.THAI_LETTERS }),
      footers: { default: pageNumberFooter() }, children: frontSection },
    // Chapters onwards: arabic numbers from 1, continuing across portrait and landscape sections.
    ...bodySections.map((section, position) => ({
      properties: page(section.landscape, position === 0
        ? { start: 1, formatType: NumberFormat.DECIMAL } : { formatType: NumberFormat.DECIMAL }),
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
