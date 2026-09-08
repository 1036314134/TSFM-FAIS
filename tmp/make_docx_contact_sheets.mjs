import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const moduleRoot = process.env.TSFM_NODE_MODULES;
if (!moduleRoot) {
  throw new Error("TSFM_NODE_MODULES is required");
}
const sharp = require(path.join(moduleRoot, "sharp"));

const [pagesDir, outputDir] = process.argv.slice(2);
if (!pagesDir || !outputDir) {
  throw new Error("Usage: make_docx_contact_sheets.mjs PAGES_DIR OUTPUT_DIR");
}

await fs.mkdir(outputDir, { recursive: true });
const files = (await fs.readdir(pagesDir))
  .filter((name) => /^page-\d+\.png$/i.test(name))
  .sort((left, right) => left.localeCompare(right, undefined, { numeric: true }));

const tileWidth = 640;
const tileHeight = 906;
const gap = 20;
const labelHeight = 34;
const columns = 2;
const rows = 2;
const canvasWidth = gap + columns * (tileWidth + gap);
const canvasHeight = gap + rows * (tileHeight + labelHeight + gap);

for (let start = 0; start < files.length; start += columns * rows) {
  const group = files.slice(start, start + columns * rows);
  const composites = [];
  for (let index = 0; index < group.length; index += 1) {
    const file = group[index];
    const column = index % columns;
    const row = Math.floor(index / columns);
    const left = gap + column * (tileWidth + gap);
    const top = gap + row * (tileHeight + labelHeight + gap);
    const pageBuffer = await sharp(path.join(pagesDir, file))
      .resize({ width: tileWidth, height: tileHeight, fit: "contain", background: "white" })
      .png()
      .toBuffer();
    const label = file.match(/\d+/)?.[0] ?? "?";
    const labelBuffer = Buffer.from(
      `<svg width="${tileWidth}" height="${labelHeight}" xmlns="http://www.w3.org/2000/svg">` +
        `<rect width="100%" height="100%" fill="#E7ECF2"/>` +
        `<text x="12" y="24" font-family="Arial" font-size="20" fill="#243447">Page ${Number(label)}</text>` +
      `</svg>`
    );
    composites.push({ input: pageBuffer, left, top });
    composites.push({ input: labelBuffer, left, top: top + tileHeight });
  }
  const firstPage = start + 1;
  const lastPage = start + group.length;
  const outputName = `pages-${String(firstPage).padStart(2, "0")}-${String(lastPage).padStart(2, "0")}.png`;
  await sharp({
    create: {
      width: canvasWidth,
      height: canvasHeight,
      channels: 3,
      background: "#AEB8C2",
    },
  })
    .composite(composites)
    .png()
    .toFile(path.join(outputDir, outputName));
}

console.log(JSON.stringify({ pages: files.length, sheets: Math.ceil(files.length / 4) }));
