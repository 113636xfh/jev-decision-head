#!/usr/bin/env node
// Deterministic SVG -> PNG. Run from any directory; no browser or GPU is used.
const { Resvg } = require('@resvg/resvg-js');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const dir = __dirname;
const names = [
  'architecture', 'reliability', 'backbone-swap', 'latency-breakdown',
  'readout-layout', 'risk-coverage', 'head-depth-ablation', 'data-scaling',
];
const requested = process.argv.slice(2);
for (const name of requested) {
  if (!names.includes(name)) throw new Error(`Unknown figure: ${name}`);
}
for (const name of requested.length ? requested : names) {
  let svg = fs.readFileSync(path.join(dir, `${name}.svg`), 'utf8');
  const match = svg.match(/viewBox="0 0 ([\d.]+) ([\d.]+)"/);
  if (!match) throw new Error(`Missing viewBox: ${name}`);
  const width = 1600;
  const height = Math.round(width * Number(match[2]) / Number(match[1]));
  // Set root dimensions explicitly. Do not depend on fitTo or CSS scaling.
  svg = svg.replace(/(<svg\b[^>]*?)width="[^"]*"/, `$1width="${width}"`)
    .replace(/(<svg\b[^>]*?)height="[^"]*"/, `$1height="${height}"`);
  const png = new Resvg(svg, {
    background: '#ffffff',
    font: { loadSystemFonts: true, defaultFontFamily: 'Arial' },
  }).render().asPng();
  if (png.readUInt32BE(16) !== width || png.readUInt32BE(20) !== height) {
    throw new Error(`Unexpected rendered dimensions: ${name}`);
  }
  fs.writeFileSync(path.join(dir, `${name}.png`), png);
  console.log(`${name}: ${width}x${height} sha256=${crypto.createHash('sha256').update(png).digest('hex')}`);
}
