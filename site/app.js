const find = (id) => document.getElementById(id);
const integers = new Intl.NumberFormat('zh-CN');
const decimals = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 4 });
const count = (value) => Number.isFinite(value) ? integers.format(value) : '—';

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
  find('audit-status').textContent = ready ? '已完成数据审计' : '等待 Kaggle 数据审计';
  const checked = files.filter((file) => Number.isFinite(file.rows));
  find('audited-count').textContent = `${checked.length} / ${files.length}`;
  find('player-count').textContent = count(files.find((item) => item.file === 'players.csv')?.unique_players);
  find('row-count').textContent = checked.length ? count(checked.reduce((total, file) => total + file.rows, 0)) : '—';
  if (ready && audit.audited_at_utc) {
    const time = new Intl.DateTimeFormat('zh-CN', { dateStyle: 'medium', timeZone: 'Asia/Shanghai' });
    find('audit-date').textContent = `${time.format(new Date(audit.audited_at_utc))}更新`;
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
    find('detail-title').textContent = missing.length > shown.length ? '缺失率最高的 12 个字段' : '字段缺失率';
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
    status.textContent = '还没有已发布的实验。完成数据审计与研究方案审阅后，这里会显示可比较的样本外指标。';
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
  find('audit-status').textContent = '无法读取审计数据';
  find('audit-date').textContent = '审计清单加载失败';
});
readData('runs').then(showRuns).catch(() => {
  find('experiments-status').textContent = '无法读取实验结果。';
});
