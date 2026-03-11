#!/usr/bin/env node
import { readFile, readdir, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import os from 'node:os';

function parseArgs(argv) {
  const out = { registry: 'https://clawhub.ai', tags: ['latest'], changelog: '' };
  for (let i = 2; i < argv.length; i += 1) {
    const a = argv[i];
    const v = argv[i + 1];
    if (a === '--skill-dir') { out.skillDir = v; i += 1; }
    else if (a === '--slug') { out.slug = v; i += 1; }
    else if (a === '--name') { out.displayName = v; i += 1; }
    else if (a === '--version') { out.version = v; i += 1; }
    else if (a === '--changelog') { out.changelog = v; i += 1; }
    else if (a === '--registry') { out.registry = v; i += 1; }
    else if (a === '--tags') { out.tags = v.split(',').map(s => s.trim()).filter(Boolean); i += 1; }
  }
  return out;
}

async function walk(dir, base = dir) {
  const out = [];
  for (const name of await readdir(dir)) {
    const p = join(dir, name);
    const st = await stat(p);
    if (st.isDirectory()) out.push(...await walk(p, base));
    else out.push({ abs: p, rel: relative(base, p).replace(/\\/g, '/') });
  }
  return out;
}

async function getToken() {
  const candidates = [
    join(os.homedir(), 'Library', 'Application Support', 'clawhub', 'config.json'),
    join(os.homedir(), 'Library', 'Application Support', 'clawdhub', 'config.json'),
    join(os.homedir(), '.config', 'clawhub', 'config.json'),
    join(os.homedir(), '.config', 'clawdhub', 'config.json'),
  ];
  for (const cfgPath of candidates) {
    if (!existsSync(cfgPath)) continue;
    try {
      const cfg = JSON.parse(await readFile(cfgPath, 'utf8'));
      if (cfg.token) return cfg.token;
    } catch {}
  }
  throw new Error('Missing clawhub token. Run `clawhub login`.');
}

const args = parseArgs(process.argv);
if (!args.skillDir || !args.slug || !args.displayName || !args.version) {
  console.error('Usage: node scripts/publish_clawhub_manual.mjs --skill-dir <dir> --slug <slug> --name <displayName> --version <X.Y.Z> [--changelog <text>]');
  process.exit(2);
}
const skillDir = resolve(args.skillDir);
const files = await walk(skillDir);
if (!files.some(f => /^skill\.md$/i.test(f.rel))) throw new Error('SKILL.md missing');
const token = await getToken();
const form = new FormData();
form.set('payload', JSON.stringify({
  slug: args.slug,
  displayName: args.displayName,
  version: args.version,
  changelog: args.changelog || '',
  tags: args.tags,
  acceptLicenseTerms: true,
}));
for (const file of files) {
  const bytes = await readFile(file.abs);
  form.append('files', new Blob([bytes], { type: 'text/plain' }), file.rel);
}
const res = await fetch(new URL('/api/v1/skills', args.registry), {
  method: 'POST',
  headers: { Accept: 'application/json', Authorization: `Bearer ${token}` },
  body: form,
});
const text = await res.text();
console.log(`HTTP ${res.status}`);
console.log(text);
if (!res.ok) process.exit(1);
