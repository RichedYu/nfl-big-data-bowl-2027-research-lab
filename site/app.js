const find = (id) => document.getElementById(id);
const english = new URLSearchParams(location.search).get('lang') === 'en';
const locale = english ? 'en' : 'zh-CN';
const t = (zh, en) => english ? en : zh;
const integers = new Intl.NumberFormat(locale);
const decimals = new Intl.NumberFormat(locale, { maximumFractionDigits: 4 });
const count = (value) => Number.isFinite(value) ? integers.format(value) : '—';

if (english) {
  document.documentElement.lang = locale;
  document.title = 'NFL Big Data Bowl 2027 · Research Dashboard';
  document.querySelector('meta[name="description"]').content = 'Data quality and experiment results for NFL Big Data Bowl 2027.';
  for (const node of document.querySelectorAll('[data-en]')) node.textContent = node.dataset.en;
  for (const node of document.querySelectorAll('[data-aria-en]')) node.setAttribute('aria-label', node.dataset.ariaEn);
  const link = find('language');
  link.textContent = '中文';
  link.lang = 'zh-CN';
}
const language = find('language');
function languageLink() {
  const url = new URL(location.href);
  if (english) url.searchParams.delete('lang');
  else url.searchParams.set('lang', 'en');
  language.href = url.href;
}
languageLink();
window.addEventListener('hashchange', languageLink);

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function row(cells) {
  const tr = element('tr');
  for (const value of cells) {
    const td = element('td');
    if (typeof value === 'string') td.textContent = value;
    else td.append(value);
    tr.append(td);
  }
  return tr;
}

function bar(label, value, max, display = decimals.format(value)) {
  const meter = element('progress');
  meter.max = max;
  meter.value = value;
  meter.setAttribute('aria-label', `${label} ${display}`);
  const wrapper = element('div', undefined, 'bar');
  wrapper.append(element('span', label), meter, element('span', display));
  return wrapper;
}

async function readData(name) {
  const response = await fetch(`./${name}.json`, { cache: 'no-cache' });
  if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
  return response.json();
}

function showAudit(audit) {
  const { files } = audit;
  const ready = audit.status === 'audited';
  const status = {
    audited: t('已完成数据审计', 'Data audit complete'),
    partial: t('已检查部分文件，跨表核验待完成', 'File checks in progress · Cross-file checks pending'),
  };
  find('audit-status').textContent = status[audit.status] ?? t('等待 Kaggle 数据审计', 'Kaggle audit pending');
  const checked = files.filter((file) => Number.isFinite(file.rows));
  find('audited-count').textContent = `${checked.length} / ${files.length}`;
  find('player-count').textContent = count(files.find((item) => item.file === 'players.csv')?.unique_players);
  find('row-count').textContent = checked.length ? count(checked.reduce((total, file) => total + file.rows, 0)) : '—';
  if (ready && audit.audited_at_utc) {
    const time = new Intl.DateTimeFormat(locale, { dateStyle: 'medium', timeZone: 'Asia/Shanghai' });
    const date = time.format(new Date(audit.audited_at_utc));
    find('audit-date').textContent = t(`${date}更新`, `Updated ${date}`);
  }
  find('inventory-body').replaceChildren(...files.map((file) => row([
    element('code', file.file), count(file.rows), count(file.fields?.length), count(file.unique_players),
  ])));
  if (checked.length === 0) return;
  const select = find('file-select');
  select.replaceChildren(...checked.map((file) => new Option(file.file, file.file)));
  const renderFields = () => {
    const file = checked.find((item) => item.file === select.value);
    const missing = (file.fields ?? []).filter((field) => Number.isFinite(field.missing_pct));
    missing.sort((first, second) => second.missing_pct - first.missing_pct || first.name.localeCompare(second.name));
    const shown = missing.slice(0, 12);
    find('detail-title').textContent = missing.length > shown.length
      ? t('缺失率最高的 12 个字段', '12 fields with the most missing values')
      : t('字段缺失率', 'Missing values by field');
    find('fields-empty').hidden = shown.length > 0;
    find('missing-chart').replaceChildren(...shown.map((field) =>
      bar(field.name, field.missing_pct, 100, `${decimals.format(field.missing_pct)}%`)));
  };
  select.addEventListener('change', renderFields);
  renderFields();
  find('audit-detail').hidden = false;
}

function showRuns({ runs }) {
  const groups = new Map();
  for (const run of runs) {
    for (const [metric, value] of Object.entries(run.metrics)) {
      if (!Number.isFinite(value)) continue;
      const group = `${metric} · ${run.comparison}`;
      if (!groups.has(group)) groups.set(group, []);
      groups.get(group).push({ ...run, value });
    }
  }
  const status = find('experiments-status');
  if (groups.size === 0) {
    status.textContent = t('尚无已发布的实验结果。', 'No experiment results have been published.');
    return;
  }
  status.hidden = true;
  const select = find('metric-select');
  select.replaceChildren(...Array.from(groups.keys(), (name) => new Option(name, name)));
  const renderMetric = () => {
    const results = groups.get(select.value);
    const max = Math.max(0, ...results.map((result) => result.value));
    const chart = find('metrics-chart');
    // Signed metrics remain in the table rather than using a zero-based progress bar.
    chart.replaceChildren(...(max > 0 && results.every((result) => result.value >= 0)
      ? results.map((result) => bar(`${result.model} · ${result.id}`, result.value, max)) : []));
    find('experiments-body').replaceChildren(...results.map((result) => row([
      result.id, result.model, result.split, count(result.n_players), decimals.format(result.value),
    ])));
  };
  select.addEventListener('change', renderMetric);
  renderMetric();
  find('experiments-content').hidden = false;
}

readData('audit').then(showAudit).catch(() => {
  find('audit-status').textContent = t('无法读取审计数据', 'Audit data unavailable');
  find('audit-date').textContent = t('审计清单加载失败', 'The file inventory could not load');
});
readData('runs').then(showRuns).catch(() => {
  find('experiments-status').textContent = t('无法读取实验结果。', 'Experiment results could not load.');
});

const canvas = find('trajectory');
const frame = find('frame');
const projection = find('projection');
const context = canvas.getContext('2d');
const track = Array.from({ length: 51 }, (_, i) => [
  12 + i * 1.57,
  14 + i * .37 + 7 * Math.sin(i / 7),
]);

function drawTrack() {
  const index = Number(frame.value);
  find('elapsed').textContent = `${(index / 10).toFixed(1)} s`;
  if (!context) return;
  const oblique = projection.value === 'perspective';
  const project = ([x, y]) => oblique ? [65 + 5.1 * x + 1.5 * y, 69 + .65 * x + 3.5 * y] : [45 + 6.3 * x, 24 + 5.7 * y];
  function line(points, color, width) {
    context.strokeStyle = color;
    context.lineWidth = width;
    context.beginPath();
    points.map(project).forEach(([x, y], i) => i === 0 ? context.moveTo(x, y) : context.lineTo(x, y));
    context.stroke();
  }
  context.clearRect(0, 0, canvas.width, canvas.height);
  line([[0, 0], [100, 0], [100, 53], [0, 53], [0, 0]], '#4f7062', 2);
  for (let x = 10; x < 100; x += 10) line([[x, 0], [x, 53]], '#2b493d', 1);
  line(track, '#638378', 2);
  line(track.slice(0, index + 1), '#c8f27c', 4);
  const [x, y] = project(track[index]);
  const [nx, ny] = project(track[Math.min(index + 1, 50)]);
  const [px, py] = project(track[Math.max(index - 1, 0)]);
  const heading = Math.atan2(ny - py, nx - px);
  context.fillStyle = '#c8f27c';
  context.beginPath();
  context.arc(x, y, 7, 0, Math.PI * 2);
  context.fill();
  context.beginPath();
  context.moveTo(x + 24 * Math.cos(heading), y + 24 * Math.sin(heading));
  context.lineTo(x + 10 * Math.cos(heading - .55), y + 10 * Math.sin(heading - .55));
  context.lineTo(x + 10 * Math.cos(heading + .55), y + 10 * Math.sin(heading + .55));
  context.fill();
}
frame.addEventListener('input', drawTrack);
projection.addEventListener('change', drawTrack);
drawTrack();
