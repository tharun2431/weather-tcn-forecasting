import re

def main():
    with open('pwa/index.html', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. ENHANCE CSS
    new_css = """
    :root {
      --bg-primary: #05050A;
      --text-primary: #FFFFFF;
      --text-secondary: rgba(255, 255, 255, 0.7);
      --text-tertiary: rgba(255, 255, 255, 0.4);
      --accent-blue: #0A84FF;
      --accent-purple: #BF5AF2;
      --accent-pink: #FF375F;
      
      --bg-card: rgba(30, 30, 35, 0.4);
      --border-light: rgba(255, 255, 255, 0.12);
      --border-dark: rgba(255, 255, 255, 0.02);
      --safe-bottom: env(safe-area-inset-bottom, 20px);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg-primary);
      color: var(--text-primary);
      overflow-x: hidden;
      user-select: none;
      overscroll-behavior: none;
    }

    /* Ambient Dynamic Background */
    .bg-gradient {
      position: fixed; inset: 0; z-index: -2;
      background: linear-gradient(to bottom, #11111C 0%, #05050A 100%);
      transition: background 1.5s ease-in-out;
    }
    .bg-mesh {
      position: fixed; top: -50%; left: -50%; width: 200%; height: 200%; z-index: -1;
      background: radial-gradient(circle at 50% 30%, rgba(10,132,255,0.15) 0%, transparent 40%),
                  radial-gradient(circle at 80% 60%, rgba(191,90,242,0.15) 0%, transparent 40%);
      animation: rotateMesh 60s linear infinite;
      pointer-events: none;
      transition: opacity 1.5s ease-in-out;
    }
    @keyframes rotateMesh { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

    .app-container {
      position: relative; z-index: 1;
      max-width: 480px; margin: 0 auto;
      padding: 0 20px 100px;
      min-height: 100vh;
    }

    /* Header */
    .header {
      display: flex; justify-content: space-between; align-items: center;
      padding: 24px 0 10px;
    }
    .location {
      font-size: 18px; font-weight: 500; font-family: "SF Pro Display", sans-serif;
      display: flex; align-items: center; gap: 6px; cursor: pointer; letter-spacing: -0.3px;
    }
    .location svg { width: 14px; height: 14px; opacity: 0.6; flex-shrink: 0; }
    .live-badge {
      display: flex; align-items: center; gap: 6px;
      background: rgba(10,132,255,0.15); border: 1px solid rgba(10,132,255,0.3);
      padding: 4px 10px; border-radius: 20px;
      font-size: 10px; font-weight: 700; color: var(--accent-blue);
      letter-spacing: 1px;
    }
    .live-badge .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent-blue); animation: pulse 2s infinite; }

    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(10,132,255,0.4); } 70% { box-shadow: 0 0 0 6px rgba(10,132,255,0); } 100% { box-shadow: 0 0 0 0 rgba(10,132,255,0); } }

    /* Main Weather */
    .main-weather {
      text-align: center; padding: 40px 0 60px;
      display: flex; flex-direction: column; align-items: center;
    }
    .main-temp {
      font-size: 110px; font-weight: 200; font-family: "SF Pro Display", sans-serif;
      line-height: 1; letter-spacing: -3px; margin-left: 20px;
      background: linear-gradient(180deg, #FFFFFF 0%, rgba(255,255,255,0.6) 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      filter: drop-shadow(0 10px 20px rgba(0,0,0,0.3));
    }
    .main-temp sup { font-size: 40px; top: -50px; position: relative; margin-left: -5px; color: rgba(255,255,255,0.6); -webkit-text-fill-color: rgba(255,255,255,0.8); }
    .weather-desc { font-size: 24px; font-weight: 500; color: var(--text-secondary); margin-bottom: 8px; letter-spacing: -0.5px; }
    .temp-range { font-size: 18px; font-weight: 500; color: var(--text-tertiary); letter-spacing: 0.5px; }

    /* Premium Standard Card */
    .card {
      background: var(--bg-card);
      backdrop-filter: blur(25px) saturate(200%); -webkit-backdrop-filter: blur(25px) saturate(200%);
      border-radius: 28px;
      padding: 20px;
      margin-bottom: 16px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      box-shadow: inset 0 1px 1px rgba(255,255,255,0.05),
                  0 8px 32px rgba(0,0,0,0.3);
    }
    .card-title {
      font-size: 13px; font-weight: 600; color: var(--text-tertiary); margin-bottom: 16px;
      display: flex; align-items: center; gap: 6px; text-transform: uppercase; letter-spacing: 1px;
    }
    .card-title svg { width: 14px; height: 14px; flex-shrink: 0; opacity: 0.8; }

    /* AI Inference Neon Card */
    .predict-card {
      position: relative; overflow: hidden;
      padding: 22px; border-radius: 28px; margin-bottom: 16px;
      backdrop-filter: blur(30px) saturate(200%); -webkit-backdrop-filter: blur(30px) saturate(200%);
      background: linear-gradient(135deg, rgba(30, 30, 35, 0.6), rgba(20, 20, 25, 0.8));
      border: 1px solid rgba(255, 255, 255, 0.1);
      box-shadow: 0 10px 40px rgba(0,0,0,0.5), inset 0 1px 1px rgba(255,255,255,0.1);
    }
    .predict-card::before { /* Super Shimmer */
      content: ''; position: absolute; top: 0; left: -100%; width: 50%; height: 100%;
      background: linear-gradient(to right, transparent, rgba(191,90,242,0.15), transparent);
      transform: skewX(-20deg); animation: shimmerLoop 4s infinite;
    }
    .predict-content { display: flex; align-items: center; gap: 18px; position: relative; z-index: 2; }
    .predict-icon {
      width: 44px; height: 44px; flex-shrink: 0;
      background: linear-gradient(135deg, rgba(10,132,255,0.2), rgba(191,90,242,0.2));
      border: 1px solid rgba(191,90,242,0.3);
      border-radius: 14px; display: flex; align-items: center; justify-content: center;
      box-shadow: 0 0 15px rgba(191,90,242,0.3);
    }
    .predict-icon svg { width: 22px; height: 22px; filter: drop-shadow(0 0 6px rgba(191,90,242,0.8)); }
    .predict-text { flex: 1; }
    .predict-text h4 {
      font-size: 15px; font-weight: 600; font-family: "SF Pro Display", sans-serif;
      background: linear-gradient(90deg, #0A84FF, #BF5AF2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      margin-bottom: 4px; letter-spacing: -0.2px;
    }
    .predict-text p { font-size: 13px; color: var(--text-secondary); line-height: 1.4; font-weight: 400;}

    /* Hourly Forecast */
    .hourly-scroll {
      display: flex; gap: 12px; overflow-x: auto; scroll-snap-type: x mandatory;
      scrollbar-width: none; padding-bottom: 5px; margin: 0 -5px; padding: 0 5px;
    }
    .hourly-scroll::-webkit-scrollbar { display: none; }
    .hourly-item {
      flex: 0 0 64px; scroll-snap-align: start; text-align: center;
      padding: 16px 4px; border-radius: 32px;
      display: flex; flex-direction: column; align-items: center; gap: 12px;
    }
    .hourly-item.now {
      background: rgba(10,132,255,0.15); border: 1px solid rgba(10,132,255,0.3);
    }
    .h-time { font-size: 14px; font-weight: 500; color: var(--text-secondary); }
    .h-icon svg { width: 28px; height: 28px; filter: drop-shadow(0 2px 5px rgba(0,0,0,0.5)); }
    .h-temp { font-size: 18px; font-weight: 500; color: var(--text-primary); }

    /* 7-Day Forecast */
    .daily-item {
      display: flex; align-items: center; padding: 12px 0;
      border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .daily-item:last-child { border-bottom: none; padding-bottom: 0; }
    .d-day { width: 50px; font-size: 16px; font-weight: 500; }
    .d-icon { width: 40px; text-align: center; }
    .d-icon svg { width: 24px; height: 24px; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.5)); }
    .d-temps { display: flex; align-items: center; gap: 10px; flex: 1; margin-left: 15px; justify-content: flex-end;}
    .temp-min { color: var(--text-tertiary); font-weight: 500; font-size: 16px; width: 28px; text-align: right;}
    .temp-bar {
      flex: 1; height: 6px; background: rgba(0,0,0,0.3); border-radius: 3px;
      position: relative; overflow: hidden;
      box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);
    }
    .temp-fill {
      position: absolute; height: 100%; border-radius: 3px;
      background: linear-gradient(90deg, #0A84FF, #BF5AF2);
    }
    .temp-max { color: var(--text-primary); font-weight: 500; font-size: 16px; width: 28px; text-align: right;}

    /* Detail Grid (Apple Weather style squares) */
    .detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px; }
    .detail-card {
      background: var(--bg-card);
      backdrop-filter: blur(25px) saturate(200%); -webkit-backdrop-filter: blur(25px) saturate(200%);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 24px; padding: 16px;
      display: flex; flex-direction: column;
      box-shadow: inset 0 1px 1px rgba(255,255,255,0.05), 0 8px 24px rgba(0,0,0,0.2);
    }
    .d-label {
      font-size: 12px; font-weight: 600; color: var(--text-tertiary);
      display: flex; align-items: center; gap: 6px; margin-bottom: 12px; letter-spacing: 0.5px; text-transform: uppercase;
    }
    .d-label svg { width: 14px; height: 14px; opacity: 0.8; }
    .d-value { font-size: 28px; font-weight: 500; color: var(--text-primary); margin-bottom: 4px; font-family: "SF Pro Display", sans-serif;}
    .d-unit { font-size: 16px; color: var(--text-secondary); font-weight: 400;}
    .d-note { font-size: 13px; font-weight: 400; color: var(--text-secondary); margin-top: auto; padding-top: 10px; line-height: 1.3;}

    /* Bottom Nav */
    .bottom-nav {
      position: fixed; bottom: 0; left: 0; right: 0;
      background: rgba(10, 10, 15, 0.65); backdrop-filter: blur(40px) saturate(200%); -webkit-backdrop-filter: blur(40px) saturate(200%);
      border-top: 1px solid rgba(255,255,255,0.1);
      display: flex; justify-content: space-around; align-items: center;
      padding: 12px 10px calc(12px + var(--safe-bottom));
      z-index: 100;
    }
    .nav-item {
      display: flex; flex-direction: column; align-items: center; gap: 6px;
      color: var(--text-tertiary); font-size: 10px; font-weight: 500;
      transition: all 0.3s; cursor: pointer; padding: 4px 12px;
    }
    .nav-item svg { width: 26px; height: 26px; transition: transform 0.3s; stroke-width: 1.5; }
    .nav-item.active { color: var(--text-primary); }
    .nav-item.active svg { color: var(--accent-blue); transform: translateY(-2px); filter: drop-shadow(0 0 10px rgba(10,132,255,0.4)); stroke-width: 2; }
    """

    # 2. Extract HTML template
    # Replace the predict-card
    new_predict_card = """    <div class="predict-card">
      <div class="predict-content">
        <div class="predict-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
            <circle cx="12" cy="12" r="4"/>
          </svg>
        </div>
        <div class="predict-text">
          <h4>Edge AI Inference</h4>
          <p id="predText">Running Neural Network locally on device...</p>
        </div>
      </div>
    </div>"""

    # Replace detail grid emojis in javascript
    # We will find the `const details = [` block and replace it.
    js_details = """
      const details = [
        { icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"/></svg>', label: 'FEELS LIKE', value: Math.round(realTemp)+ '°', note: 'Wind feels cooler' },
        { icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z"/></svg>', label: 'HUMIDITY', value: Math.round(rh)+ '%', note: 'Current moisture level' },
        { icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2"/></svg>', label: 'WIND', value: wv.toFixed(1)+ '<span class="d-unit"> km/h</span>', note: 'Gentle breeze' },
        { icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/></svg>', label: 'PRESSURE', value: Math.round(pressure)+ '<span class="d-unit"> hPa</span>', note: pressure > 1020 ? 'High pressure system' : pressure < 1000 ? 'Low pressure system' : 'Stable atmospheric pressure' },
        { icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>', label: 'UV INDEX', value: uvIndex.toFixed(0), note: uvLevel, customStyle: `color:${uvColor}` },
        { icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>', label: 'SUN', value: '<div style="font-size:16px;">Sunrise: ' + sunrise + '</div>', note: '<div style="font-size:16px;">Sunset: ' + sunset + '</div>' }
      ];
    """

    # We use regex to substitute the CSS and JS and HTML!
    
    # 1. Substitute CSS
    content = re.sub(r'<style>.*?</style>', f'<style>\n{new_css}\n</style>', content, flags=re.DOTALL)
    
    # 2. Substitute Predict Card
    content = re.sub(r'<div class="predict-card">.*?</div>\n    </div>', new_predict_card, content, flags=re.DOTALL)
    # The above regex might fail if inner divs didn't match. Let's do a more robust replacement.
    content = re.sub(r'<div class="predict-card">.*?<p id="predText">.*?</p>\s*</div>\s*</div>', new_predict_card, content, flags=re.DOTALL)

    # 3. Substitute details array
    content = re.sub(r'const details = \[.*?\n\s*\];', js_details, content, flags=re.DOTALL)
    
    # 4. We also need to fix the JS mapping of the details array to HTML because we added SVG and customStyle
    detail_html_script = """
  document.getElementById('detailGrid').innerHTML = details.map(d => `
    <div class="detail-card">
      <div class="d-label">${d.icon} <span>${d.label}</span></div>
      <div class="d-value" style="${d.customStyle || ''}">${d.value}</div>
      <div class="d-note">${d.note}</div>
    </div>
  `).join('');
"""
    content = re.sub(r"document\.getElementById\('detailGrid'\)\.innerHTML = details\.map\(.*?`\)\.join\(''\);", detail_html_script.strip(), content, flags=re.DOTALL)

    # Also map emoji in SVG for hourly icons in JS 'getWeatherInfo(code)'
    new_weather_info_js = """
function getWeatherIcon(code) {
  // Simplified SVG premium icons
  if (code <= 1) return '<svg viewBox="0 0 24 24" fill="none" stroke="#FFD60A" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>'; // Sun
  if (code <= 3) return '<svg viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/></svg>'; // Cloud
  if (code <= 49) return '<svg viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/><path d="M8 20v2M12 20v2M16 20v2"/></svg>'; // Fog/Drizzle
  if (code <= 69) return '<svg viewBox="0 0 24 24" fill="none" stroke="#0A84FF" stroke-width="2"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/><path d="M8 22l-2-2M12 22l-2-2M16 22l-2-2"/></svg>'; // Rain
  if (code <= 79) return '<svg viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/><path d="M8 23l-1-1M12 23l-1-1M16 23l-1-1"/></svg>'; // Snow
  return '<svg viewBox="0 0 24 24" fill="none" stroke="#BF5AF2" stroke-width="2"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/><path d="M13 22l-3-6 4-2-6-8"/></svg>'; // Thunder
}

function getWeatherInfo(code) {
  let icon = getWeatherIcon(code);
  let desc = 'Clear';
  if (code == 1 || code == 2) desc = 'Partly Cloudy';
  if (code == 3) desc = 'Overcast';
  if (code >= 45 && code <= 48) desc = 'Fog';
  if (code >= 51 && code <= 67) desc = 'Rain';
  if (code >= 71 && code <= 77) desc = 'Snow';
  if (code >= 95) desc = 'Thunderstorm';
  return { icon, desc };
}
"""
    content = re.sub(r'function getWeatherInfo\(code\) \{.*?\return \{ icon: icon, desc: desc \};\n\}', new_weather_info_js.strip(), content, flags=re.DOTALL)

    # In updateWeatherUI, the SVG output needs to be injected using innerHTML instead of textContent
    content = content.replace("document.getElementById('mainIcon').textContent = current.icon;", "document.getElementById('mainIcon').innerHTML = current.icon;")

    # Fix hourly loop output
    # old: `<div class="h-icon">${info.icon}</div>` => this is already innerHTML so SVGs just work directly!
    # Fix daily forecast output: `<div class="d-icon">${info.icon}</div>` => this is already innerHTML so SVGs work.

    # 5. Fix the search icon SVG sizing
    content = content.replace('.search-wrap svg { position: absolute; left: 18px; top: 18px; width: 22px; height: 22px; color: var(--text-tertiary); }',
                              '.search-wrap svg { position: absolute; left: 18px; top: 18px; width: 22px; height: 22px; color: var(--text-tertiary); }')


    # Save
    with open('pwa/index.html', 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("UI Upgraded Successfully")

if __name__ == "__main__":
    main()
