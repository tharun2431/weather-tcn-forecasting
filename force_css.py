import shutil

with open('pwa/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

new_css = """
    *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

    :root {
      /* Dynamic Theme Variables */
      --bg-primary: #0a0a0f;
      --bg-gradient-1: #1a1b26;
      --bg-gradient-2: #0f0f1a;
      
      /* Glassmorphism Next-Gen */
      --bg-card: rgba(255, 255, 255, 0.03);
      --bg-card-hover: rgba(255, 255, 255, 0.08);
      --border-light: rgba(255, 255, 255, 0.12);
      --border-dark: rgba(255, 255, 255, 0.02);
      --shadow-glass: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
      
      /* Text */
      --text-primary: #ffffff;
      --text-secondary: rgba(255, 255, 255, 0.7);
      --text-tertiary: rgba(255, 255, 255, 0.4);
      
      /* Accents */
      --accent-blue: #38bdf8;
      --accent-purple: #c084fc;
      --accent-pink: #f472b6;
      --accent-green: #4ade80;
      
      --safe-top: env(safe-area-inset-top, 0px);
      --safe-bottom: env(safe-area-inset-bottom, 0px);
    }

    /* Clean Body Base */
    html { height: 100%; background: #000; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg-primary);
      color: var(--text-primary);
      min-height: 100%;
      overflow-x: hidden;
      -webkit-font-smoothing: antialiased;
      padding-top: var(--safe-top);
      padding-bottom: var(--safe-bottom);
      transition: background 1.5s ease;
    }

    /* ===== ULTRA DYNAMIC MESH BACKGROUND ===== */
    .bg-mesh {
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      z-index: 0;
      pointer-events: none;
      background: radial-gradient(circle at top left, var(--bg-gradient-1), transparent 60%),
                  radial-gradient(circle at bottom right, var(--bg-gradient-2), transparent 60%);
      transition: background 1.5s ease;
    }
    
    .bg-mesh::after {
      content: '';
      position: absolute;
      top: -50%; left: -50%;
      width: 200%; height: 200%;
      background: radial-gradient(circle at center, rgba(56, 189, 248, 0.05) 0%, transparent 40%);
      animation: fluidMotion 15s infinite linear;
      opacity: var(--aura-opacity, 1);
      transition: opacity 1s ease;
    }

    @keyframes fluidMotion {
      0%   { transform: rotate(0deg) translate(2%, 2%); }
      50%  { transform: rotate(180deg) translate(-2%, -2%) scale(1.1); }
      100% { transform: rotate(360deg) translate(2%, 2%); }
    }

    /* Layout */
    .app-container {
      position: relative;
      z-index: 1;
      max-width: 430px;
      margin: 0 auto;
      padding: 20px 16px 100px;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    /* Header */
    .header {
      display: flex; justify-content: space-between; align-items: center;
      margin-bottom: 22px; padding: 0 4px;
    }
    .location {
      font-size: 16px; font-weight: 600; color: var(--text-secondary);
      display: flex; align-items: center; gap: 8px; cursor: pointer;
    }
    .location:hover { color: var(--text-primary); }
    
    /* Live Edge AI Badge */
    .live-badge {
      display: flex; align-items: center; gap: 6px;
      padding: 6px 12px; border-radius: 20px;
      background: rgba(0,0,0,0.4);
      border: 1px solid var(--border-light);
      font-size: 11px; font-weight: 800; color: var(--text-secondary);
      letter-spacing: 0.5px;
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      box-shadow: 0 4px 15px rgba(0,0,0,0.2);
      transition: all 0.3s ease;
    }
    .live-badge.live {
      border-color: rgba(74, 222, 128, 0.4);
      color: var(--accent-green);
      box-shadow: 0 0 20px rgba(74, 222, 128, 0.15);
    }
    .live-badge .dot {
      width: 6px; height: 6px; border-radius: 50%;
      background: currentColor;
    }
    .live-badge.live .dot { animation: pulseGreen 2s infinite; }
    
    @keyframes pulseGreen {
      0% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.7); }
      70% { box-shadow: 0 0 0 6px rgba(74, 222, 128, 0); }
      100% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0); }
    }

    /* Main Weather Block */
    .main-weather {
      text-align: center; padding: 10px 0 30px;
      animation: floatIn 0.8s cubic-bezier(0.2, 0.8, 0.2, 1);
    }
    .weather-icon {
      font-size: 110px; line-height: 1; filter: drop-shadow(0 15px 35px rgba(0,0,0,0.5));
      margin-bottom: 2px;
      animation: iconFloat 4s ease-in-out infinite;
    }
    @keyframes iconFloat {
      0%, 100% { transform: translateY(0); }
      50% { transform: translateY(-8px); }
    }
    .main-temp {
      font-size: 110px; font-weight: 200; letter-spacing: -7px; line-height: 1;
      color: #fff; text-shadow: 0 10px 40px rgba(255,255,255,0.15);
      margin-left: -5px;
    }
    .main-temp sup { font-size: 45px; font-weight: 300; vertical-align: super; }
    .weather-desc { font-size: 22px; font-weight: 500; color: var(--text-secondary); text-transform: capitalize; margin-top: 0px;}
    .temp-range { font-size: 16px; color: var(--text-tertiary); margin-top: 10px; font-weight: 500; }
    .temp-range span { color: var(--text-primary); }

    /* ===== NEXT GEN GLASSMORPHISM CARDS ===== */
    .card {
      background: var(--bg-card);
      border-radius: 24px;
      padding: 20px;
      margin-bottom: 16px;
      backdrop-filter: blur(40px) saturate(150%);
      -webkit-backdrop-filter: blur(40px) saturate(150%);
      box-shadow: var(--shadow-glass);
      border: 1px solid transparent;
      background-image: linear-gradient(var(--bg-card), var(--bg-card)), linear-gradient(135deg, var(--border-light), var(--border-dark));
      background-origin: border-box;
      background-clip: padding-box, border-box;
      transition: transform 0.2s cubic-bezier(0.3, 0.7, 0.4, 1.5), background 0.3s;
    }
    .card:hover { background: var(--bg-card-hover); }

    .card-title {
      font-size: 13px; font-weight: 700; color: var(--text-tertiary); margin-bottom: 16px;
      display: flex; align-items: center; gap: 8px; text-transform: uppercase; letter-spacing: 1.5px;
    }

    /* ===== AI METALLIC SHIMMER CARD ===== */
    .predict-card {
      position: relative; overflow: hidden;
      background: linear-gradient(145deg, rgba(56, 189, 248, 0.1), rgba(192, 132, 252, 0.05));
      border: 1px solid rgba(192, 132, 252, 0.2);
    }
    .predict-card::before { /* Shimmer */
      content: ''; position: absolute; top: 0; left: -100%; width: 50%; height: 100%;
      background: linear-gradient(to right, transparent, rgba(255,255,255,0.1), transparent);
      transform: skewX(-20deg);
      animation: shimmerLoop 5s infinite;
    }
    @keyframes shimmerLoop {
      0% { left: -100%; }
      15% { left: 200%; }
      100% { left: 200%; }
    }
    
    .predict-content { display: flex; align-items: center; gap: 16px; position: relative; z-index: 2; }
    .predict-icon {
      width: 48px; height: 48px; flex-shrink: 0;
      background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
      border-radius: 14px; display: flex; align-items: center; justify-content: center;
      font-size: 24px; box-shadow: 0 4px 20px rgba(192, 132, 252, 0.4);
    }
    .predict-text h4 { font-size: 15px; font-weight: 700; color: var(--accent-blue); margin-bottom: 4px; text-shadow: 0 0 10px rgba(56, 189, 248, 0.3); }
    .predict-text p { font-size: 13px; color: var(--text-secondary); line-height: 1.5; font-weight: 500;}

    /* ===== HOURLY SCROLL ===== */
    .hourly-scroll {
      display: flex; gap: 10px; overflow-x: auto; scroll-snap-type: x mandatory;
      scrollbar-width: none; padding-bottom: 5px; margin: 0 -5px; padding: 0 5px;
    }
    .hourly-scroll::-webkit-scrollbar { display: none; }
    
    .hourly-item {
      flex: 0 0 68px; scroll-snap-align: start; text-align: center;
      padding: 16px 4px; border-radius: 34px;
      background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.03);
    }
    .hourly-item.now {
      background: linear-gradient(180deg, rgba(56,189,248,0.25), rgba(56,189,248,0.05));
      border-color: rgba(56,189,248,0.3);
      box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .hourly-item.ai-prediction {
      background: linear-gradient(180deg, rgba(244,114,182,0.15), rgba(0,0,0,0.2));
      border-color: rgba(244,114,182,0.25);
      position: relative; overflow: hidden;
    }
    .hourly-item.ai-prediction::before {
      content: ''; position: absolute; inset: 0; background: radial-gradient(circle at top, rgba(244,114,182,0.1), transparent);
    }
    .h-time { font-size: 13px; font-weight: 600; color: var(--text-secondary); margin-bottom: 10px; }
    .h-icon { font-size: 26px; filter: drop-shadow(0 2px 8px rgba(0,0,0,0.6)); margin-bottom: 10px; }
    .h-temp { font-size: 18px; font-weight: 700; color: var(--text-primary); }
    .hourly-item.ai-prediction .h-temp { color: var(--accent-pink); text-shadow: 0 0 10px rgba(244,114,182,0.5); }

    /* ===== 7 DAY FORECAST CAPSULES ===== */
    .daily-item {
      display: flex; align-items: center; justify-content: space-between;
      padding: 14px 0; border-bottom: 1px solid var(--border-dark);
    }
    .daily-item:last-child { border-bottom: none; padding-bottom: 0; }
    .d-day { width: 50px; font-size: 16px; font-weight: 600; color: var(--text-primary); }
    .d-day.today { color: var(--accent-blue); }
    .d-icon { width: 35px; font-size: 24px; text-align: center; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.4)); }
    
    .d-temps { display: flex; align-items: center; gap: 10px; flex: 1; margin-left: 15px; justify-content: flex-end;}
    .d-lo { width: 30px; text-align: right; font-size: 16px; font-weight: 600; color: var(--text-tertiary); }
    
    .d-bar-bg {
      flex: 1; height: 8px; background: rgba(0,0,0,0.4); border-radius: 8px;
      position: relative; overflow: hidden; margin: 0 10px;
      box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
    }
    .d-bar-fill {
      position: absolute; top: 0; height: 100%; border-radius: 8px;
      background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
      box-shadow: inset -2px 0 2px rgba(255,255,255,0.5);
    }
    .d-hi { width: 30px; text-align: right; font-size: 16px; font-weight: 600; color: var(--text-primary); }

    /* ===== DETAIL GRID ===== */
    .detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .detail-card {
      background: var(--bg-card); border-radius: 20px; padding: 18px;
      border: 1px solid transparent;
      background-image: linear-gradient(var(--bg-card), var(--bg-card)), linear-gradient(135deg, var(--border-light), var(--border-dark));
      background-origin: border-box; background-clip: padding-box, border-box;
      display: flex; flex-direction: column; justify-content: space-between; min-height: 120px;
    }
    .detail-card .d-label { font-size: 12px; font-weight: 700; color: var(--text-tertiary); margin-bottom: 12px; display: flex; align-items: center; gap: 6px;}
    .detail-card .d-value { font-size: 28px; font-weight: 600; color: var(--text-primary); letter-spacing: -0.5px;}
    .detail-card .d-unit { font-size: 16px; color: var(--text-secondary); font-weight: 500;}
    .detail-card .d-note { font-size: 13px; font-weight: 500; color: var(--text-secondary); margin-top: auto; padding-top: 12px; line-height:1.4;}

    /* Navigation Bar */
    .nav-bar {
      position: fixed; bottom: 0; left: 0; right: 0;
      background: rgba(5, 5, 10, 0.7); backdrop-filter: blur(40px) saturate(150%); -webkit-backdrop-filter: blur(40px) saturate(150%);
      border-top: 1px solid var(--border-light);
      display: flex; justify-content: space-around; align-items: center;
      padding: 12px 10px calc(12px + var(--safe-bottom));
      z-index: 100;
    }
    .nav-item {
      display: flex; flex-direction: column; align-items: center; gap: 6px;
      color: var(--text-tertiary); font-size: 11px; font-weight: 600;
      transition: all 0.3s; cursor: pointer; padding: 4px 12px; border-radius: 12px;
    }
    .nav-item svg { width: 24px; height: 24px; transition: transform 0.3s; }
    .nav-item.active { color: var(--text-primary); }
    .nav-item.active svg { color: var(--accent-blue); transform: translateY(-4px); filter: drop-shadow(0 4px 10px rgba(56,189,248,0.6));}

    /* Screens & Overlays */
    .screen { display: none; animation: fadeIn 0.4s cubic-bezier(0.2, 0.8, 0.2, 1); }
    .screen.active { display: block; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(15px); filter: blur(5px);} to { opacity: 1; transform: translateY(0); filter: blur(0);} }

    .search-input {
      width: 100%; background: rgba(0,0,0,0.3); border: 2px solid rgba(255,255,255,0.1);
      border-radius: 20px; padding: 18px; color: white; font-size: 18px; font-family: inherit; font-weight: 500;
      outline: none; transition: border-color 0.3s, background 0.3s, box-shadow 0.3s; margin-bottom: 24px;
      backdrop-filter: blur(20px);
    }
    .search-input:focus { border-color: var(--accent-blue); background: rgba(0,0,0,0.5); box-shadow: 0 0 20px rgba(56,189,248,0.2);}
    
    .city-item {
      padding: 18px; border-bottom: 1px solid var(--border-dark);
      display: flex; justify-content: space-between; align-items: center; cursor: pointer;
      border-radius: 16px; margin-bottom: 8px; transition: background 0.2s;
    }
    .city-item:active { background: rgba(255,255,255,0.08); }
    .city-name { font-size: 20px; font-weight: 600; color: white; }
    .city-country { font-size: 14px; color: var(--text-secondary); margin-top: 6px; font-weight: 500;}

    /* Premium Loading Screen */
    .loading-screen {
      position: fixed; top: 0; left: 0; width: 100%; height: 100%;
      background: var(--bg-primary); z-index: 999;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
    }
    .loader-orb { width: 100px; height: 100px; position: relative; margin-bottom: 40px; }
    .orb-ring {
      position: absolute; inset: 0; border-radius: 50%;
      border: 3px solid transparent; border-top-color: var(--accent-blue);
      animation: spin 1.5s cubic-bezier(0.68, -0.55, 0.265, 1.55) infinite;
    }
    .orb-ring:nth-child(2) { border-top-color: var(--accent-purple); animation-direction: reverse; animation-duration: 2s; }
    .loader-icon { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font-size: 40px; }
    .brand-title {
      font-size: 28px; font-weight: 800; letter-spacing: 3px; text-transform: uppercase;
      background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
      -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 30px rgba(56,189,248,0.3);
    }
    @keyframes spin { 100% { transform: rotate(360deg); } }
"""

try:
    prefix = html.split('<style>')[0]
    suffix = html.split('</style>')[1]
    
    final_html = prefix + '<style>\n' + new_css + '\n  </style>' + suffix
    
    # Inject bg-mesh and dynamic theme if they are completely missing
    if '<div class="bg-mesh"' not in final_html:
        final_html = final_html.replace('<body>', '<body>\n<div class="bg-mesh" id="bgMesh"></div>')
        final_html = final_html.replace("function getWeatherInfo(code) {", """
function updateWeatherTheme(code) {
  const root = document.documentElement;
  if ([0, 1].includes(code)) {
    root.style.setProperty('--bg-gradient-1', '#0f2027');
    root.style.setProperty('--bg-gradient-2', '#203a43');
    root.style.setProperty('--aura-opacity', '1');
  } else if ([2, 3, 45, 48].includes(code)) {
    root.style.setProperty('--bg-gradient-1', '#1e293b');
    root.style.setProperty('--bg-gradient-2', '#0f172a');
    root.style.setProperty('--aura-opacity', '0.4');
  } else if ([51,53,55,61,63,65,80,81,82].includes(code)) {
    root.style.setProperty('--bg-gradient-1', '#111827');
    root.style.setProperty('--bg-gradient-2', '#1e1b4b');
    root.style.setProperty('--aura-opacity', '0.2');
  } else {
    root.style.setProperty('--bg-gradient-1', '#000000');
    root.style.setProperty('--bg-gradient-2', '#171717');
    root.style.setProperty('--aura-opacity', '0.8');
  }
}
function getWeatherInfo(code) {
""")
        final_html = final_html.replace("const weatherInfo = getWeatherInfo(c.weather_code);", "const weatherInfo = getWeatherInfo(c.weather_code);\n  updateWeatherTheme(c.weather_code);")
        final_html = final_html.replace("document.getElementById('aiStatusText').textContent = \"EDGE AI LIVE\";", "document.getElementById('aiStatusText').textContent = \"EDGE AI LIVE\";\n      document.getElementById('aiStatusBadge').classList.add('live');")

    with open('pwa/index.html', 'w', encoding='utf-8') as f:
        f.write(final_html)
    print("SUCCESS: Forced the Premium CSS into pwa/index.html")
except Exception as e:
    print("Error:", e)
