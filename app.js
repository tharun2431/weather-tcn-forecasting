/* DeepWeather — multi-city neural forecasting
   LSTM runs in-browser via ONNX Runtime. Conformal intervals from the project's
   per-horizon calibration. Live conditions from Open-Meteo. */

const CITIES = [
  { id:'jena',   name:'Jena',     country:'Germany',   lat:50.9227, lon:11.5865, mean:10.1, lo:-19.7, hi:34.3 },
  { id:'london', name:'London',   country:'UK',        lat:51.5074, lon:-0.1278, mean:10.8, lo:-7.8,  hi:34.7 },
  { id:'ny',     name:'New York', country:'USA',       lat:40.7128, lon:-74.0060,mean:11.6, lo:-20.7, hi:37.4 },
  { id:'sydney', name:'Sydney',   country:'Australia', lat:-33.8688,lon:151.2093,mean:17.3, lo:3.3,   hi:38.1 },
  { id:'tokyo',  name:'Tokyo',    country:'Japan',     lat:35.6762, lon:139.6503,mean:14.8, lo:-9.1,  hi:37.0 },
];

// conformal 90% interval half-widths, calibrated per forecast hour
const Q_HAT = [1.259,1.548,1.844,2.153,2.422,2.659,2.859,3.046,3.208,3.358,
               3.501,3.650,3.780,3.909,4.020,4.134,4.232,4.320,4.405,4.500,
               4.575,4.657,4.745,4.811];

// all tuned by grid search on validation data, so the comparison is like for like
const MODELS = [
  { name:'TFT (this project)',  mae:1.247, ours:true  },
  { name:'Gradient boosting',   mae:1.263, ours:false },
  { name:'LSTM (this project)', mae:1.266, ours:true  },
  { name:'Linear regression',   mae:1.575, ours:false },
  { name:'Seasonal naive',      mae:2.091, ours:false },
  { name:'Climatology',         mae:2.470, ours:false },
  { name:'Persistence',         mae:2.935, ours:false },
];

const WMO = {0:'Clear',1:'Mainly clear',2:'Partly cloudy',3:'Overcast',45:'Fog',48:'Rime fog',
  51:'Light drizzle',53:'Drizzle',55:'Heavy drizzle',61:'Light rain',63:'Rain',65:'Heavy rain',
  66:'Freezing rain',67:'Freezing rain',71:'Light snow',73:'Snow',75:'Heavy snow',77:'Snow grains',
  80:'Light showers',81:'Showers',82:'Heavy showers',85:'Snow showers',86:'Snow showers',
  95:'Thunderstorm',96:'Thunderstorm, hail',99:'Thunderstorm, hail'};

const FEATURE_COUNT = 14, SEQ_LEN = 168, HORIZON = 24;

let session = null, scaler = null, current = CITIES[0], wx = null, fcData = null;

// measure from the wrapper - the <svg> itself can report 0/stale before layout settles
const chartWidth = svg => Math.round(
  (svg.parentElement && svg.parentElement.getBoundingClientRect().width) || svg.clientWidth || 640
);

/* ── model ─────────────────────────────────────────────── */
async function initModel(){
  const chip = document.getElementById('modelChip');
  const state = document.getElementById('modelState');
  try{
    scaler = await (await fetch('scaler.json')).json();
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
    session = await ort.InferenceSession.create('lstm_model.onnx');
    chip.classList.add('ready'); state.textContent = 'neural net ready';
    return true;
  }catch(e){
    console.error('model load failed', e);
    chip.classList.add('err'); state.textContent = 'model unavailable';
    return false;
  }
}

// 14 derived features, matching the training pipeline exactly
function featuresAt(h, i){
  const T = h.temperature_2m[i], p = h.surface_pressure[i];
  const rh = h.relative_humidity_2m[i], Tdew = h.dew_point_2m[i];
  let wv = h.wind_speed_10m[i]/3.6; if(!(wv>=0)) wv = 0;
  let mx = h.wind_gusts_10m[i]/3.6; if(!(mx>=0)) mx = wv;
  const wd = h.wind_direction_10m[i];
  const TK = T+273.15;
  const Tpot = TK*Math.pow(1000/p,0.286);
  const VPmax = 6.112*Math.exp(17.67*T/(T+243.5));
  const VPact = VPmax*(rh/100);
  const VPdef = VPmax-VPact;
  const sh = 622*VPact/(p-0.378*VPact);
  const H2OC = VPact/p*1000;
  const Tv = TK*(1+sh/1000*0.61);
  const rho = p*100/(287.05*Tv)*1000;
  const raw = [p,T,Tpot,Tdew,rh,VPmax,VPact,VPdef,sh,H2OC,rho,wv,mx,wd];
  return raw.map((v,k)=>(v-scaler.mean[k])/scaler.scale[k]);
}

async function forecast(h, nowIdx){
  if(!session||!scaler) return null;
  const out = [];
  for(let s=0;s<HORIZON;s++){
    const start = nowIdx-(SEQ_LEN-1)+s;
    if(start<0 || start+SEQ_LEN>h.temperature_2m.length) break;
    const seq = [];
    for(let i=start;i<start+SEQ_LEN;i++) seq.push(featuresAt(h,i));
    const t = new ort.Tensor('float32', Float32Array.from(seq.flat()), [1,SEQ_LEN,FEATURE_COUNT]);
    const r = await session.run({ input:t });
    const scaled = r.temperature.data[0];
    out.push(scaled*scaler.scale[1]+scaler.mean[1]);   // index 1 = T (degC)
  }
  return out;
}

/* ── data ──────────────────────────────────────────────── */
async function loadCity(city){
  const u = `https://api.open-meteo.com/v1/forecast?latitude=${city.lat}&longitude=${city.lon}`
    + `&current=temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m,pressure_msl,weather_code`
    + `&hourly=temperature_2m,weather_code,relative_humidity_2m,dew_point_2m,surface_pressure,`
    + `wind_speed_10m,wind_direction_10m,wind_gusts_10m`
    + `&daily=temperature_2m_max,temperature_2m_min&timezone=auto&past_days=8&forecast_days=3`;
  const r = await fetch(u);
  if(!r.ok) throw new Error('weather fetch failed');
  return r.json();
}

const nowIndex = d => {
  const t = d.current.time.slice(0,13);
  const i = d.hourly.time.findIndex(x=>x.slice(0,13)===t);
  return i>=0 ? i : d.hourly.time.length-1;
};

/* ── render: hero + stats ──────────────────────────────── */
// little count-up on the big number, feels less abrupt than it snapping in
function countUp(el, to, ms=650){
  // rAF does not run in a hidden/background tab, so skip the animation there
  // and just show the number - otherwise it would sit on the placeholder
  if(matchMedia('(prefers-reduced-motion: reduce)').matches || document.hidden){
    el.textContent = to; return;
  }
  const from = 0, t0 = performance.now();
  const step = now => {
    const k = Math.min(1, (now-t0)/ms);
    const e = 1-Math.pow(1-k,3);
    el.textContent = Math.round(from + (to-from)*e);
    if(k<1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
  // belt and braces: guarantee the final value lands even if frames never fire
  setTimeout(()=>{ el.textContent = to; }, ms+150);
}

function renderNow(city,d){
  document.getElementById('heroLoading').hidden = true;
  const body = document.getElementById('heroBody');
  body.hidden = false;
  body.classList.remove('rise'); void body.offsetWidth; body.classList.add('rise');
  const c = d.current, t = Math.round(c.temperature_2m);
  document.getElementById('heroCity').textContent = `${city.name}, ${city.country}`;
  document.getElementById('heroTime').textContent =
    new Date(c.time).toLocaleString([], {weekday:'long', hour:'2-digit', minute:'2-digit'});
  document.getElementById('heroCond').textContent = WMO[c.weather_code] ?? '—';
  countUp(document.getElementById('heroTemp'), t);
  // past_days shifts the daily arrays back, so find TODAY rather than index 0
  const today = c.time.slice(0,10);
  const di = Math.max(0, d.daily.time.indexOf(today));
  document.getElementById('heroRange').textContent =
    `High ${Math.round(d.daily.temperature_2m_max[di])}° · Low ${Math.round(d.daily.temperature_2m_min[di])}°`;
  // hero glow shifts cold→warm
  const k = Math.max(0, Math.min(1, (t+10)/45));
  document.getElementById('hero').style.setProperty('--hero-glow',
    `rgba(${Math.round(57+(217-57)*k)},${Math.round(135+(89-135)*k)},${Math.round(229+(38-229)*k)},.17)`);

  const stats = [
    ['Feels like', Math.round(c.apparent_temperature), '°'],
    ['Humidity',   Math.round(c.relative_humidity_2m), '%'],
    ['Wind',       Math.round(c.wind_speed_10m), ' km/h'],
    ['Pressure',   Math.round(c.pressure_msl), ' hPa'],
  ];
  document.getElementById('statGrid').innerHTML = stats.map(([l,v,u])=>
    `<div class="stat"><div class="stat-lab">${l}</div><div class="stat-val">${v}<small>${u}</small></div></div>`).join('');
}

/* ── render: forecast chart ────────────────────────────── */
function renderForecast(d, nowIdx, pred){
  const svg = document.getElementById('fcChart');
  const W = chartWidth(svg), H = 260;
  const M = {t:14, r:14, b:26, l:34};
  const iw = W-M.l-M.r, ih = H-M.t-M.b;

  const api = [], labels = [];
  for(let i=1;i<=HORIZON;i++){
    api.push(d.hourly.temperature_2m[nowIdx+i]);
    labels.push(new Date(d.hourly.time[nowIdx+i]).getHours());
  }
  const n = pred ? Math.min(pred.length, HORIZON) : HORIZON;
  fcData = {pred, api, labels, n};

  const lows = [], highs = [];
  for(let i=0;i<n;i++){
    if(pred){ lows.push(pred[i]-Q_HAT[i]); highs.push(pred[i]+Q_HAT[i]); }
    lows.push(api[i]); highs.push(api[i]);
  }
  let lo = Math.min(...lows.filter(Number.isFinite));
  let hi = Math.max(...highs.filter(Number.isFinite));
  const pad = Math.max(1,(hi-lo)*0.12); lo-=pad; hi+=pad;

  const X = i => M.l + (n<2?0:i/(n-1))*iw;
  const Y = v => M.t + (1-(v-lo)/(hi-lo))*ih;

  // y ticks
  const ticks = 4, parts = [];
  for(let k=0;k<=ticks;k++){
    const v = lo+(hi-lo)*k/ticks, y = Y(v);
    parts.push(`<line class="grid-line" x1="${M.l}" y1="${y}" x2="${W-M.r}" y2="${y}"/>`);
    parts.push(`<text class="axis-txt" x="${M.l-7}" y="${y+3.5}" text-anchor="end">${Math.round(v)}°</text>`);
  }
  // x ticks (every 4h)
  for(let i=0;i<n;i+=4){
    parts.push(`<text class="axis-txt" x="${X(i)}" y="${H-8}" text-anchor="middle">${String(labels[i]).padStart(2,'0')}</text>`);
  }

  // conformal band
  if(pred){
    const up = [], dn = [];
    for(let i=0;i<n;i++){ up.push(`${X(i)},${Y(pred[i]+Q_HAT[i])}`); dn.push(`${X(i)},${Y(pred[i]-Q_HAT[i])}`); }
    parts.push(`<polygon class="ser-band" points="${up.join(' ')} ${dn.reverse().join(' ')}"/>`);
  }
  const path = arr => arr.map((v,i)=>`${i?'L':'M'}${X(i)},${Y(v)}`).join(' ');
  parts.push(`<path class="ser-line" d="${path(api.slice(0,n))}" stroke="var(--s2)" stroke-dasharray="5 4"/>`);
  if(pred) parts.push(`<path id="fcLine" class="ser-line" d="${path(pred.slice(0,n))}" stroke="var(--s1)"/>`);

  parts.push(`<line id="fcCross" class="crosshair" y1="${M.t}" y2="${M.t+ih}" style="display:none"/>`);
  parts.push(`<circle id="fcDot1" class="hover-dot" r="4.5" fill="var(--s1)" style="display:none"/>`);
  parts.push(`<circle id="fcDot2" class="hover-dot" r="4.5" fill="var(--s2)" style="display:none"/>`);
  parts.push(`<rect id="fcHit" x="${M.l}" y="${M.t}" width="${iw}" height="${ih}" fill="transparent"/>`);

  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  svg.setAttribute('height',H);
  svg.innerHTML = parts.join('');

  // animate the neural line drawing itself in (skipped if reduced-motion)
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  if(pred && !reduce){
    const ln = svg.querySelector('#fcLine');
    const band = svg.querySelector('.ser-band');
    if(ln){
      const len = ln.getTotalLength();
      ln.style.setProperty('--len', len);
      ln.classList.add('draw');
    }
    if(band) band.classList.add('fade');
  }
  wireHover(svg, X, Y, n, M, iw, ih);

  document.getElementById('fcNote').textContent = pred
    ? 'The interval widens with lead time because uncertainty grows — calibrated per hour by conformal prediction.'
    : 'Neural forecast unavailable, showing the numerical weather model only.';
}

function wireHover(svg,X,Y,n,M,iw,ih){
  const tip = document.getElementById('fcTip');
  const cross = svg.querySelector('#fcCross');
  const d1 = svg.querySelector('#fcDot1'), d2 = svg.querySelector('#fcDot2');
  const hit = svg.querySelector('#fcHit');
  const show = on => {
    [cross,d1,d2].forEach(el=>{ if(el) el.style.display = on?'':'none'; });
    tip.hidden = !on;
  };
  const move = ev => {
    const r = svg.getBoundingClientRect();
    const px = (ev.touches?ev.touches[0].clientX:ev.clientX) - r.left;
    const sx = px * (svg.viewBox.baseVal.width / r.width);
    let i = Math.round((sx-M.l)/(iw/Math.max(1,n-1)));
    i = Math.max(0, Math.min(n-1, i));
    const {pred,api,labels} = fcData;
    cross.setAttribute('x1',X(i)); cross.setAttribute('x2',X(i));
    if(pred){ d1.setAttribute('cx',X(i)); d1.setAttribute('cy',Y(pred[i])); }
    else d1.style.display='none';
    d2.setAttribute('cx',X(i)); d2.setAttribute('cy',Y(api[i]));
    show(true);
    const rows = [];
    if(pred){
      rows.push(`<div class="tip-r"><span class="k" style="background:var(--s1)"></span><span class="n">Neural net</span><span class="v">${pred[i].toFixed(1)}°</span></div>`);
      rows.push(`<div class="tip-r"><span class="k" style="background:var(--band);border:1px solid rgba(57,135,229,.6)"></span><span class="n">90% range</span><span class="v">${(pred[i]-Q_HAT[i]).toFixed(1)}–${(pred[i]+Q_HAT[i]).toFixed(1)}°</span></div>`);
    }
    rows.push(`<div class="tip-r"><span class="k" style="background:var(--s2)"></span><span class="n">Weather model</span><span class="v">${api[i].toFixed(1)}°</span></div>`);
    tip.innerHTML = `<div class="tip-h">+${i+1}h · ${String(labels[i]).padStart(2,'0')}:00</div>${rows.join('')}`;
    const wrap = svg.parentElement.getBoundingClientRect();
    const cx = X(i) * (r.width / svg.viewBox.baseVal.width);
    tip.style.left = Math.max(72, Math.min(wrap.width-72, cx)) + 'px';
    tip.style.top = Math.max(58, (pred?Y(pred[i]):Y(api[i])) * (r.height/svg.viewBox.baseVal.height) - 12) + 'px';
  };
  hit.addEventListener('mousemove',move);
  hit.addEventListener('touchmove',e=>{move(e);e.preventDefault();},{passive:false});
  hit.addEventListener('mouseleave',()=>show(false));
  hit.addEventListener('touchend',()=>show(false));
}

/* ── render: model chart ───────────────────────────────── */
function renderModels(){
  const svg = document.getElementById('mdChart');
  const W = chartWidth(svg);
  const rowH = 34, M = {t:6, r:46, b:40, l:132};
  const H = M.t + MODELS.length*rowH + M.b;
  const iw = W-M.l-M.r;
  const max = 3.2;
  const parts = [];

  const axisY = M.t + MODELS.length*rowH - 8;
  for(const v of [0,1,2,3]){
    const x = M.l + v/max*iw;
    parts.push(`<line class="grid-line" x1="${x}" y1="${M.t}" x2="${x}" y2="${axisY}"/>`);
    parts.push(`<text class="axis-txt" x="${x}" y="${axisY+15}" text-anchor="middle">${v}</text>`);
  }
  // label the axis - a chart with bare numbers and no unit is unreadable
  parts.push(`<text class="axis-txt" x="${M.l+iw/2}" y="${axisY+32}" text-anchor="middle">Mean absolute error (°C)</text>`);
  MODELS.forEach((m,i)=>{
    const y = M.t + i*rowH, bh = 18;
    const w = m.mae/max*iw;
    const fill = m.ours ? 'var(--s1)' : 'rgba(255,255,255,.16)';
    parts.push(`<text class="axis-txt" x="${M.l-9}" y="${y+bh/2+3.5}" text-anchor="end">${m.name}</text>`);
    parts.push(`<rect class="md-bar" data-i="${i}" x="${M.l}" y="${y}" width="${w}" height="${bh}" rx="4" fill="${fill}"/>`);
    parts.push(`<text class="axis-txt" x="${M.l+w+7}" y="${y+bh/2+3.5}" style="fill:var(--ink-2)">${m.mae.toFixed(2)}</text>`);
  });

  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  svg.setAttribute('height',H);
  svg.innerHTML = parts.join('');

  const tip = document.getElementById('mdTip');
  svg.querySelectorAll('.md-bar').forEach(bar=>{
    bar.style.cursor='pointer';
    bar.addEventListener('mouseenter',e=>{
      const m = MODELS[+bar.dataset.i];
      tip.hidden = false;
      tip.innerHTML = `<div class="tip-h">${m.name}</div><div class="tip-r"><span class="n">Test MAE</span><span class="v">${m.mae.toFixed(2)} °C</span></div>`;
      const r = svg.getBoundingClientRect(), wrap = svg.parentElement.getBoundingClientRect();
      const br = bar.getBoundingClientRect();
      tip.style.left = Math.min(wrap.width-80, br.right-r.left+40)+'px';
      tip.style.top  = (br.top-r.top+10)+'px';
    });
    bar.addEventListener('mouseleave',()=>{tip.hidden=true;});
  });

  document.getElementById('modelTable').innerHTML =
    '<thead><tr><th>Model</th><th style="text-align:right">Test MAE (°C)</th></tr></thead><tbody>'
    + MODELS.map(m=>`<tr><td>${m.name}</td><td class="num">${m.mae.toFixed(2)}</td></tr>`).join('')
    + '</tbody>';
}

/* ── render: cities ────────────────────────────────────── */
function renderCityTabs(){
  document.getElementById('cityRow').innerHTML = CITIES.map(c=>
    `<button class="city-tab${c.id===current.id?' is-active':''}" data-id="${c.id}" role="tab">${c.name}</button>`).join('');
  document.querySelectorAll('.city-tab').forEach(b=>
    b.addEventListener('click',()=>selectCity(CITIES.find(c=>c.id===b.dataset.id))));
}

let cityTemps = {};
function renderCityGrid(){
  document.getElementById('cityGrid').innerHTML = CITIES.map(c=>{
    const t = cityTemps[c.id];
    return `<div class="citycard${c.id===current.id?' is-active':''}" data-id="${c.id}">
      <div class="cc-name">${c.name}</div>
      <div class="cc-meta">${c.country}</div>
      <div class="cc-temp">${t==null?'—':Math.round(t)+'°'}</div>
      <div class="cc-band">avg ${c.mean}° · range ${c.lo}° to ${c.hi}°</div>
    </div>`;
  }).join('');
  document.querySelectorAll('.citycard').forEach(el=>
    el.addEventListener('click',()=>selectCity(CITIES.find(c=>c.id===el.dataset.id))));
}

/* five-city small multiples - each city's next 24h on its own scale, so the
   comparison is about the SHAPE of the day rather than who is warmest */
let cityCurves = {};

function renderSparks(){
  document.getElementById('sparkGrid').innerHTML = CITIES.map(c=>{
    const s = cityCurves[c.id];
    if(!s || !s.temps || s.temps.length<2){
      return `<div class="spark" data-id="${c.id}"><div class="sp-top"><span class="sp-name">${c.name}</span></div>
              <div class="skel skel-line" style="height:46px;margin-top:9px"></div></div>`;
    }
    const t = s.temps, W = 150, Hh = 46;
    const lo = Math.min(...t), hi = Math.max(...t), rng = Math.max(0.5, hi-lo);
    const X = i => i/(t.length-1)*W;
    const Y = v => 4 + (1-(v-lo)/rng)*(Hh-8);
    const line = t.map((v,i)=>`${i?'L':'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(' ');
    const area = `${line} L${W},${Hh} L0,${Hh} Z`;
    return `<div class="spark${c.id===current.id?' is-active':''}" data-id="${c.id}">
      <div class="sp-top">
        <span class="sp-name">${c.name}</span>
        <span class="sp-now">${Math.round(t[0])}°</span>
      </div>
      <div class="sp-sub">${c.country}</div>
      <svg class="sp-svg" viewBox="0 0 ${W} ${Hh}" preserveAspectRatio="none" aria-hidden="true">
        <path class="sp-area" d="${area}"/><path class="sp-path" d="${line}"/>
      </svg>
      <div class="sp-range"><span>${Math.round(lo)}°</span><span>swing ${Math.round(hi-lo)}°</span><span>${Math.round(hi)}°</span></div>
    </div>`;
  }).join('');
  document.querySelectorAll('.spark').forEach(el=>
    el.addEventListener('click',()=>selectCity(CITIES.find(c=>c.id===el.dataset.id))));
}

async function loadAllCityTemps(){
  renderSparks();   // skeletons first
  await Promise.all(CITIES.map(async c=>{
    try{
      const r = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${c.lat}&longitude=${c.lon}`
        + `&current=temperature_2m&hourly=temperature_2m&timezone=auto&forecast_days=2`);
      const j = await r.json();
      cityTemps[c.id] = j.current.temperature_2m;
      const k = j.hourly.time.findIndex(x=>x.slice(0,13)===j.current.time.slice(0,13));
      const s = k>=0 ? k : 0;
      cityCurves[c.id] = { temps: j.hourly.temperature_2m.slice(s, s+24) };
    }catch{ cityTemps[c.id] = null; cityCurves[c.id] = null; }
  }));
  renderCityGrid();
  renderSparks();
}

/* ── flow ──────────────────────────────────────────────── */
async function selectCity(city){
  if(!city) return;
  current = city;
  renderCityTabs();
  renderCityGrid();   // re-render so the highlighted card always matches the tab
  renderSparks();
  document.getElementById('heroBody').hidden = true;
  document.getElementById('heroLoading').hidden = false;
  document.getElementById('heroLoading').textContent = `Loading ${city.name}…`;
  try{
    wx = await loadCity(city);
    const i = nowIndex(wx);
    renderNow(city,wx);
    renderForecast(wx,i,null);                 // draw immediately
    const pred = await forecast(wx.hourly,i);  // then the neural pass
    if(pred && pred.length) renderForecast(wx,i,pred);
  }catch(e){
    console.error(e);
    document.getElementById('heroLoading').textContent = 'Could not load conditions. Check your connection.';
  }
}

function wireNav(){
  const btns = [...document.querySelectorAll('.nav-btn')];
  btns.forEach(b=>b.addEventListener('click',()=>{
    document.getElementById(b.dataset.target)
      .scrollIntoView({behavior:'smooth',block:'start'});
  }));
  const secs = ['now','forecast','cities','models'].map(id=>document.getElementById(id));
  const io = new IntersectionObserver(es=>{
    es.forEach(e=>{
      if(e.isIntersecting){
        btns.forEach(b=>b.classList.toggle('is-active', b.dataset.target===e.target.id));
      }
    });
  },{rootMargin:'-45% 0px -50% 0px'});
  secs.forEach(s=>io.observe(s));
}

// re-draw charts whenever their container actually has a width (covers first
// layout, font load, orientation change and resize - no fragile timing guesses)
let lastW = {fc:0, md:0};
function observeCharts(){
  const ro = new ResizeObserver(entries=>{
    for(const e of entries){
      const w = Math.round(e.contentRect.width);
      if(w < 40) continue;
      if(e.target.id==='fcWrap' && w!==lastW.fc){
        lastW.fc = w;
        if(wx && fcData) renderForecast(wx, nowIndex(wx), fcData.pred);
      }
      if(e.target.id==='mdWrap' && w!==lastW.md){
        lastW.md = w;
        renderModels();
      }
    }
  });
  const fcW = document.getElementById('fcChart').parentElement;
  const mdW = document.getElementById('mdChart').parentElement;
  fcW.id = 'fcWrap'; mdW.id = 'mdWrap';
  ro.observe(fcW); ro.observe(mdW);
}

(async function init(){
  renderCityTabs();
  wireNav();
  renderModels();         // draw once now...
  observeCharts();        // ...and redraw whenever the width actually changes
  loadAllCityTemps();
  await initModel();
  await selectCity(current);
  // offline caching only when actually deployed - a service worker on localhost
  // just serves stale files while developing
  const isLocal = ['localhost','127.0.0.1'].includes(location.hostname);
  if('serviceWorker' in navigator && !isLocal){
    navigator.serviceWorker.register('sw.js').catch(()=>{});
  }
})();
