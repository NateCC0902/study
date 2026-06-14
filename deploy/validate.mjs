#!/usr/bin/env node
/* Validate the unified study site: structure, offline-safety, bilingual parity.
   Run:  node deploy/validate.mjs            (exits non-zero on any issue)
   Flags: --course=pid|mldl|embedded   --lang=en|zh   --skip-missing            */
import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const ROOT = new URL('..', import.meta.url).pathname;
const ARG = Object.fromEntries(process.argv.slice(2).map(a => { const m = a.match(/^--([^=]+)(?:=(.*))?$/); return m ? [m[1], m[2] ?? true] : [a, true]; }));

/* expected pages, mirroring assets/app.js COURSES (to also catch MISSING files) */
const EMBED = {
  arduino: { dir: 'embedded-mastery/Arduino', ids: ['01-intro','02-gpio','03-analog','04-pwm','05-timers-interrupts','06-serial-i2c-spi','07-sensors-actuators','08-architecture-power','09-capstone'] },
  stm32:   { dir: 'embedded-mastery/STM32',   ids: ['01-intro','02-clocks','03-gpio','04-nvic-exti','05-timers','06-adc-dac','07-uart-i2c-spi','08-dma','09-rtos-lowpower','10-capstone'] },
};
const MLDL_IDS = ['00-what-is-ml','01-math-foundations','02-linear-regression','03-gradient-descent','04-logistic-regression','05-overfitting-evaluation','06-knn-trees-ensembles','07-svm-kernels','08-kmeans-pca','09-neural-networks-mlp','10-backpropagation','11-pytorch-fundamentals','12-training-deep-nets','13-cnns','14-rnns-lstms','15-attention-transformers','16-generative-models','17-transfer-learning-llms-mlops','18-reinforcement-learning','19-detection-segmentation','20-data-feature-engineering','21-hyperparameter-optimization','appendix','glossary'];
const PID_IDS = ['01-plant','02-p-control','03-d-control','04-i-control','05-windup','06-derivative-noise','07-wrapping-actuators','08-tuning-ros2','glossary'];
const IO_IDS = ['01-io-model','02-devices-sysfs','03-udev-naming','04-drivers-modules-firmware','05-buses-enumeration','06-serial-tty','07-interrupts-dma-mmio','08-multiplexing-epoll','09-networking-io','10-can-socketcan','11-realtime-robust-io','12-capstone-robot-io','glossary'];
const REF_PAGES = new Set(['appendix','glossary']);
const ML_INTERACTIVE = { '02-linear-regression':1,'03-gradient-descent':1,'04-logistic-regression':1,'05-overfitting-evaluation':1,'08-kmeans-pca':2,'09-neural-networks-mlp':1,'13-cnns':1,'15-attention-transformers':1 };
const IO_INTERACTIVE = { '05-buses-enumeration':1,'06-serial-tty':1,'07-interrupts-dma-mmio':1,'08-multiplexing-epoll':1,'09-networking-io':1,'10-can-socketcan':1 };

const pages = [];
const push = (rel, meta) => pages.push({ rel, ...meta });
for (const [trk, t] of Object.entries(EMBED)) for (const id of t.ids) {
  push(`${t.dir}/${id}.html`,        { course:'embedded', track:trk, chapter:id, lang:'en', isRef:false });
  push(`${t.dir}/zh-tw/${id}.html`,  { course:'embedded', track:trk, chapter:id, lang:'zh', isRef:false });
}
for (const id of MLDL_IDS) {
  push(`ml-dl-curriculum/${id}.html`,       { course:'mldl', track:'main', chapter:id, lang:'en', isRef:REF_PAGES.has(id) });
  push(`ml-dl-curriculum/zh-tw/${id}.html`, { course:'mldl', track:'main', chapter:id, lang:'zh', isRef:REF_PAGES.has(id) });
}
for (const id of PID_IDS) {
  push(`pid/${id}.html`,       { course:'pid', track:'main', chapter:id, lang:'en', isRef:REF_PAGES.has(id) });
  push(`pid/zh-tw/${id}.html`, { course:'pid', track:'main', chapter:id, lang:'zh', isRef:REF_PAGES.has(id) });
}
for (const id of IO_IDS) {
  push(`io/${id}.html`,       { course:'io', track:'main', chapter:id, lang:'en', isRef:REF_PAGES.has(id) });
  push(`io/zh-tw/${id}.html`, { course:'io', track:'main', chapter:id, lang:'zh', isRef:REF_PAGES.has(id) });
}

const sel = pages.filter(p => (!ARG.course || p.course === ARG.course) && (!ARG.lang || p.lang === ARG.lang));

let missing = 0, totalIssues = 0, checked = 0;
const rows = [];

for (const pg of sel) {
  const abs = join(ROOT, pg.rel);
  if (!existsSync(abs)) { if (!ARG['skip-missing']) { rows.push({ rel: pg.rel, missing: true }); missing++; } continue; }
  checked++;
  const html = readFileSync(abs, 'utf8');
  const issues = [];
  const depth = pg.rel.split('/').length - 1;
  const assetPrefix = '../'.repeat(depth) + 'assets/';

  if (!/^﻿?<!DOCTYPE html>/i.test(html)) issues.push('no <!DOCTYPE html>');
  if (!/<\/html>\s*$/i.test(html)) issues.push('no closing </html>');
  if (/\bTODO\b|lorem ipsum|\.\.\.\s*<\/(p|li|td)>/i.test(html)) issues.push('placeholder/TODO/ellipsis');
  const ext = [...html.matchAll(/(?:src|href)\s*=\s*"(https?:\/\/[^"]+)"/g)].map(m => m[1]);
  if (ext.length) issues.push('external URL: ' + ext.slice(0,2).join(', '));

  if (!new RegExp(`data-course="${pg.course}"`).test(html)) issues.push(`data-course != ${pg.course}`);
  if (!new RegExp(`data-track="${pg.track}"`).test(html)) issues.push(`data-track != ${pg.track}`);
  if (!new RegExp(`data-chapter="${pg.chapter}"`).test(html)) issues.push(`data-chapter != ${pg.chapter}`);

  for (const need of ['styles.css','app.js','vendor/mermaid.min.js','katex/katex.min.js','katex/contrib/auto-render.min.js','vendor/highlight.min.js','katex/katex.min.css']) {
    if (!html.includes(assetPrefix + need)) issues.push('missing/!depth asset: ' + assetPrefix + need);
  }
  const scripts = [...html.matchAll(/<script[^>]*src="([^"]+)"[^>]*>/g)].map(m => m[1]);
  const appIdx = scripts.findIndex(s => s.endsWith('assets/app.js'));
  if (appIdx === -1) issues.push('no app.js');
  else if (scripts.slice(appIdx + 1).some(s => /vendor|katex/.test(s))) issues.push('vendor lib after app.js');

  if (!/<nav class="sidebar" id="sidebar">\s*<\/nav>/.test(html)) issues.push('#sidebar not present-and-empty');
  if (!/<nav class="chap-nav" id="chap-nav">\s*<\/nav>/.test(html)) issues.push('#chap-nav not present-and-empty');

  const h2ids = [...html.matchAll(/<h2\b[^>]*\bid="([^"]+)"/g)].map(m => m[1]);
  const h2all = [...html.matchAll(/<h2\b/g)].length;
  const minH2 = pg.isRef ? 3 : 6;
  if (h2all < minH2) issues.push(`only ${h2all} <h2> (want >=${minH2})`);
  if (h2ids.length !== h2all) issues.push(`${h2all - h2ids.length} <h2> missing id`);
  const dupe = [...new Set(h2ids.filter((v,i) => h2ids.indexOf(v) !== i))];
  if (dupe.length) issues.push('dup h2 id: ' + dupe.join(','));

  const exTypes = [...html.matchAll(/data-ex="([^"]+)"/g)].map(m => m[1]);
  if (pg.isRef) {
    if (exTypes.length < 1) issues.push('ref page: want >=1 exercise');
  } else {
    if (exTypes.length < 3) issues.push(`only ${exTypes.length} exercises (want 3)`);
    for (const ty of ['quiz','numeric','reveal']) if (!exTypes.includes(ty)) issues.push('missing exercise type: ' + ty);
    if (!/data-correct="true"/.test(html)) issues.push('no data-correct="true"');
    if (exTypes.includes('numeric') && !/data-ex="numeric"[^>]*data-answer="/.test(html)) issues.push('numeric missing data-answer');
  }
  const exIds = [...html.matchAll(/class="exercise"[^>]*data-id="([^"]+)"/g)].map(m => m[1]);
  const dupId = [...new Set(exIds.filter((v,i) => exIds.indexOf(v) !== i))];
  if (dupId.length) issues.push('dup exercise data-id: ' + dupId.join(','));

  let badCode = 0;
  for (const m of html.matchAll(/<code\b[^>]*>([\s\S]*?)<\/code>/g)) if (/</.test(m[1])) badCode++;
  if (badCode) issues.push(`${badCode} <code> with unescaped '<'`);

  if (pg.course === 'pid' && !pg.isRef) {
    if (!html.includes(assetPrefix + 'pid-sim.js')) issues.push('pid: missing pid-sim.js');
    if (!/EM\.pidSim\(/.test(html)) issues.push('pid: missing EM.pidSim() init');
    if (!/<canvas id="/.test(html)) issues.push('pid: missing <canvas>');
  }
  if (pg.course === 'mldl' && ML_INTERACTIVE[pg.chapter]) {
    if (!html.includes(assetPrefix + 'ml-demos.js')) issues.push('ml: missing ml-demos.js');
    const canv = [...html.matchAll(/EM\.mlDemo\.\w+\(/g)].length;
    if (canv < ML_INTERACTIVE[pg.chapter]) issues.push(`ml: ${canv} demos (want ${ML_INTERACTIVE[pg.chapter]})`);
  }
  if (pg.course === 'io' && IO_INTERACTIVE[pg.chapter]) {
    if (!html.includes(assetPrefix + 'io-sim.js')) issues.push('io: missing io-sim.js');
    const canv = [...html.matchAll(/EM\.ioDemo\.\w+\(/g)].length;
    if (canv < IO_INTERACTIVE[pg.chapter]) issues.push(`io: ${canv} demos (want ${IO_INTERACTIVE[pg.chapter]})`);
  }

  rows.push({ rel: pg.rel, h2: h2all, ex: exTypes.length, issues });
  totalIssues += issues.length;
}

for (const r of rows) {
  if (r.missing) { console.log('MISSING  ' + r.rel); continue; }
  if (r.issues.length) { console.log(`✗ ${r.rel}  (h2=${r.h2} ex=${r.ex})`); for (const i of r.issues) console.log('     ↳ ' + i); }
}
const okCount = rows.filter(r => !r.missing && !r.issues.length).length;
console.log('─'.repeat(70));
console.log(`${sel.length} expected · ${okCount} ok · ${checked - okCount} with issues · ${missing} MISSING · ${totalIssues} total issue(s)`);
process.exit(missing || totalIssues ? 1 : 0);
