import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

const ROOT = new URL('..', import.meta.url).pathname;
const files = [
  'index.html',
  ...readdirSync(join(ROOT,'Arduino')).filter(f=>f.endsWith('.html')).sort().map(f=>'Arduino/'+f),
  ...readdirSync(join(ROOT,'STM32')).filter(f=>f.endsWith('.html')).sort().map(f=>'STM32/'+f),
];

let totalIssues = 0;
const rows = [];

for (const rel of files) {
  const html = readFileSync(join(ROOT, rel), 'utf8');
  const issues = [];
  const isHub = rel === 'index.html';

  // 1. doctype + closing html
  if (!/^﻿?<!DOCTYPE html>/i.test(html)) issues.push('no <!DOCTYPE html> at start');
  if (!/<\/html>\s*$/i.test(html)) issues.push('does not end with </html>');
  if (/\bTODO\b|\blorem ipsum\b|\.\.\.\s*<\/(p|li)>/i.test(html)) issues.push('possible placeholder/TODO/ellipsis');

  // 2. offline: no external URLs in src/href attributes
  const ext = [...html.matchAll(/(?:src|href)\s*=\s*"(https?:\/\/[^"]+)"/g)].map(m=>m[1]);
  if (ext.length) issues.push('external URL(s): '+ext.slice(0,3).join(', '));

  if (!isHub) {
    // body attrs
    const track = rel.startsWith('Arduino') ? 'arduino' : 'stm32';
    const tcls  = track === 'arduino' ? 'track-arduino' : 'track-stm32';
    const chap  = rel.split('/')[1].replace('.html','');
    if (!new RegExp(`<body[^>]*class="[^"]*${tcls}`).test(html)) issues.push(`body missing class ${tcls}`);
    if (!new RegExp(`data-track="${track}"`).test(html)) issues.push(`body data-track != ${track}`);
    if (!new RegExp(`data-chapter="${chap}"`).test(html)) issues.push(`body data-chapter != ${chap}`);

    // 3. empty nav containers
    if (!/<nav class="sidebar" id="sidebar">\s*<\/nav>/.test(html)) issues.push('#sidebar not present-and-empty');
    if (!/<nav class="chap-nav" id="chap-nav">\s*<\/nav>/.test(html)) issues.push('#chap-nav not present-and-empty');

    // 4. asset paths + script order
    for (const need of ['../assets/styles.css','../assets/app.js','../assets/vendor/mermaid.min.js',
                        '../assets/katex/katex.min.js','../assets/katex/contrib/auto-render.min.js',
                        '../assets/vendor/highlight.min.js','../assets/katex/katex.min.css']) {
      if (!html.includes(need)) issues.push('missing asset ref: '+need);
    }
    const scripts = [...html.matchAll(/<script[^>]*src="([^"]+)"[^>]*>/g)].map(m=>m[1]);
    const appIdx = scripts.indexOf('../assets/app.js');
    if (appIdx === -1) issues.push('no app.js script');
    else {
      // every vendor lib script must come before app.js
      const after = scripts.slice(appIdx+1).filter(s=>/vendor|katex/.test(s));
      if (after.length) issues.push('lib script after app.js: '+after.join(','));
    }

    // 5. headings
    const h2 = [...html.matchAll(/<h2\b[^>]*\bid="([^"]+)"/g)].map(m=>m[1]);
    const h2all = [...html.matchAll(/<h2\b/g)].length;
    if (h2all < 6) issues.push(`only ${h2all} <h2> (want >=6)`);
    if (h2.length !== h2all) issues.push(`${h2all-h2.length} <h2> missing id`);
    const dupe = h2.filter((v,i)=>h2.indexOf(v)!==i);
    if (dupe.length) issues.push('duplicate h2 id: '+[...new Set(dupe)].join(','));

    // 6. exercises
    const exTypes = [...html.matchAll(/data-ex="([^"]+)"/g)].map(m=>m[1]);
    if (exTypes.length < 3) issues.push(`only ${exTypes.length} exercises`);
    for (const t of ['quiz','numeric','reveal']) if (!exTypes.includes(t)) issues.push('missing exercise type: '+t);
    const exIds = [...html.matchAll(/class="exercise"[^>]*data-id="([^"]+)"/g)].map(m=>m[1]);
    const dupId = exIds.filter((v,i)=>exIds.indexOf(v)!==i);
    if (dupId.length) issues.push('dup exercise data-id: '+dupId.join(','));
    // quiz must have a correct option
    if (!/data-correct="true"/.test(html)) issues.push('no data-correct="true" option');
    // numeric must have an answer
    if (exTypes.includes('numeric') && !/data-ex="numeric"[^>]*data-answer="/.test(html)
        && !/data-answer="[^"]+"[^>]*data-ex="numeric"/.test(html)
        && !/<div class="exercise" data-ex="numeric"[\s\S]{0,200}?data-answer="/.test(html))
      issues.push('numeric exercise missing data-answer');

    // 7. richness
    const mermaid = [...html.matchAll(/<div class="mermaid">/g)].length;
    const code = [...html.matchAll(/<div class="codeblock">/g)].length;
    if (mermaid < 1) issues.push('no mermaid diagram');
    if (code < 2) issues.push(`only ${code} codeblock(s)`);

    // 8. unescaped < inside <code> ... </code>
    let badCode = 0;
    for (const m of html.matchAll(/<code\b[^>]*>([\s\S]*?)<\/code>/g)) {
      if (/</.test(m[1])) badCode++;
    }
    if (badCode) issues.push(`${badCode} code block(s) with unescaped '<'`);

    rows.push({rel, h2: h2all, ex: exTypes.length, mermaid, code, issues});
  } else {
    if (!/data-page="hub"/.test(html)) issues.push('hub missing data-page');
    if (!html.includes('assets/app.js')) issues.push('hub missing app.js');
    if (!/track-card arduino/.test(html) || !/track-card stm32/.test(html)) issues.push('hub missing track cards');
    rows.push({rel, h2:'-', ex:'-', mermaid:'-', code:'-', issues});
  }
  totalIssues += issues.length;
}

console.log('FILE                                   H2  EX  MMD CODE  STATUS');
console.log('─'.repeat(74));
for (const r of rows) {
  const status = r.issues.length ? '✗ '+r.issues.length+' issue(s)' : '✓ ok';
  console.log(r.rel.padEnd(38), String(r.h2).padStart(2), String(r.ex).padStart(3), String(r.mermaid).padStart(3), String(r.code).padStart(4), '  '+status);
  for (const i of r.issues) console.log('     ↳', i);
}
console.log('─'.repeat(74));
console.log(`${rows.length} files · ${totalIssues} total issue(s)`);
