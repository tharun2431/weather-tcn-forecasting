import re

def main():
    with open('pwa/index.html', 'r', encoding='utf-8') as f:
        content = f.read()

    # --- 1. CSS REPLACEMENT ---
    material_css = """
    :root {
      --bg-primary: #546571; /* Google Weather Slate Blue background */
      --bg-gradient-1: #6E8291;
      --bg-gradient-2: #3E4E59;
      --text-primary: #E2E2E2;
      --text-secondary: #C4C7C5;
      --text-tertiary: #8E918F;
      --bg-card: #131416; /* Material dark surface */
      --nav-bg: #1E2022;
      --safe-bottom: env(safe-area-inset-bottom, 20px);
      --font-google: 'Google Sans', -apple-system, Roboto, sans-serif;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
    body {
      font-family: var(--font-google);
      background: linear-gradient(180deg, var(--bg-gradient-1) 0%, var(--bg-gradient-2) 100%);
      background-attachment: fixed;
      color: var(--text-primary);
      overflow-x: hidden;
      user-select: none;
    }

    /* Ambient Background Mask */
    .bg-mesh, .bg-gradient { display: none; } /* Disable old glassmorphism background */

    .app-container {
      max-width: 480px; margin: 0 auto;
      padding: 0 16px 120px; min-height: 100vh;
    }

    /* Header */
    .header { padding: 40px 0 10px; display: flex; align-items: center; gap: 16px; }
    .header svg.back-btn { width: 24px; height: 24px; color: var(--text-primary); }
    .location { font-size: 24px; font-weight: 400; color: #fff; cursor: pointer; }

    /* Main Weather */
    .main-weather { text-align: center; padding: 20px 0 30px; }
    .main-temp {
      font-size: 110px; font-weight: 400; font-family: var(--font-google);
      line-height: 1; letter-spacing: -4px; margin-left: 20px; color: #fff;
    }
    .main-temp sup { font-size: 44px; position: relative; top: -50px; left: -5px; }
    .weather-desc { font-size: 20px; font-weight: 400; color: #fff; margin-bottom: 6px; }
    .temp-range { font-size: 16px; font-weight: 400; color: #fff; opacity: 0.8;}

    /* Material Cards */
    .card {
      background: var(--bg-card); border-radius: 28px;
      padding: 16px; margin-bottom: 12px;
      border: none; box-shadow: none;
    }
    .card-title {
      font-size: 14px; font-weight: 500; color: var(--text-primary); margin-bottom: 12px;
      display: flex; align-items: center; gap: 8px;
    }
    .card-title svg { width: 16px; height: 16px; }

    /* AI Widget */
    .predict-card {
      background: var(--bg-card); border-radius: 28px; padding: 16px; margin-bottom: 12px;
    }
    .ai-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .ai-title { display: flex; align-items: center; font-size: 14px; font-weight: 500; gap: 8px; }
    .ai-title svg { width: 18px; height: 18px; color: #D1E4DD; }
    .ai-summary { font-size: 15px; font-weight: 400; color: #E2E2E2; line-height: 1.4; margin-bottom: 12px; }
    .ai-bullets { list-style: none; padding-left: 10px; }
    .ai-bullets li { position: relative; font-size: 14px; color: var(--text-secondary); margin-bottom: 6px; padding-left: 12px; line-height: 1.4; }
    .ai-bullets li::before { content: '•'; position: absolute; left: 0; color: var(--text-secondary); }
    .ai-badge { display: flex; align-items: center; justify-content: center; gap: 6px; margin-top: 16px; opacity: 0.6; font-size: 12px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 12px; }

    /* Hourly Forecast (Horizontal list) */
    .hourly-scroll {
      display: flex; gap: 0px; overflow-x: auto; scroll-snap-type: x mandatory;
      scrollbar-width: none; margin: 0 -16px; padding: 0 16px;
    }
    .hourly-scroll::-webkit-scrollbar { display: none; }
    .hourly-item {
      flex: 0 0 54px; text-align: center; display: flex; flex-direction: column; align-items: center; gap: 6px;
    }
    .h-temp { font-size: 15px; font-weight: 500; color: #fff; }
    .h-icon svg { width: 24px; height: 24px; margin: 4px 0; }
    .h-precip { font-size: 11px; color: #A8C7FA; font-weight: 500; }
    .h-time { font-size: 13px; color: var(--text-secondary); margin-top: 2px;}

    /* 10-Day Forecast (Horizontal Pills) */
    .daily-scroll {
      display: flex; gap: 8px; overflow-x: auto; scroll-snap-type: x mandatory; margin: 0 -16px; padding: 0 16px;
      scrollbar-width: none;
    }
    .daily-scroll::-webkit-scrollbar { display: none; }
    .daily-pill {
      flex: 0 0 68px; background: rgba(255,255,255,0.03); border-radius: 40px; padding: 16px 8px;
      display: flex; flex-direction: column; align-items: center; border: 1px solid rgba(255,255,255,0.05);
    }
    .daily-pill.today { border: 1px solid rgba(255,255,255,0.2); background: rgba(255,255,255,0.08); }
    .d-hi { font-size: 16px; font-weight: 500; color: #fff; }
    .d-lo { font-size: 14px; font-weight: 400; color: var(--text-tertiary); margin-bottom: 8px; }
    .d-icon svg { width: 24px; height: 24px; }
    .d-precip { font-size: 12px; color: #A8C7FA; font-weight: 500; margin-top: 4px; }
    .d-day { font-size: 14px; color: var(--text-secondary); margin-top: 10px; }
    .d-date { font-size: 12px; color: var(--text-tertiary); }

    /* Weather Map */
    .map-card { padding: 0; overflow: hidden; height: 160px; position: relative;}
    .map-title { position: absolute; top: 12px; left: 16px; z-index: 2; padding: 4px 10px; background: var(--bg-card); border-radius: 12px; display: flex; align-items: center; gap: 6px; font-size: 13px; font-weight: 500;}
    .map-img { width: 100%; height: 100%; object-fit: cover; opacity: 0.8; }

    /* Details Grid CSS Shapes */
    .detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
    .m-card { background: var(--bg-card); border-radius: 28px; display: flex; flex-direction: column; padding: 16px; position: relative; overflow: hidden; }
    .m-circle { aspect-ratio: 1/1; border-radius: 50%; justify-content: center; align-items: center; text-align: center; padding: 20px;}
    
    .m-label { font-size: 13px; font-weight: 500; color: var(--text-primary); display: flex; align-items: center; gap: 6px; }
    .m-label svg { width: 16px; height: 16px; opacity: 0.7;}
    .m-val { font-size: 32px; font-weight: 400; color: #fff; margin-top: 10px;}
    .m-unit { font-size: 16px; color: var(--text-secondary);}
    .m-desc { font-size: 12px; color: var(--text-secondary); margin-top: auto;}

    /* specific shapes */
    .shape-scallop { clip-path: polygon(50% 0%, 61% 4%, 73% 2%, 82% 10%, 91% 15%, 95% 26%, 100% 36%, 98% 48%, 100% 59%, 95% 69%, 89% 79%, 80% 86%, 70% 92%, 59% 97%, 48% 99%, 37% 96%, 27% 92%, 18% 86%, 10% 77%, 5% 66%, 1% 55%, 3% 43%, 0% 32%, 5% 21%, 11% 12%, 22% 6%, 33% 2%, 43% 0%); }
    .shape-wave { position: absolute; top: 0; left: 0; right: 0; height: 25px; background: rgba(255,255,255,0.05); clip-path: polygon(0 0, 100% 0, 100% 100%, 75% 80%, 50% 100%, 25% 80%, 0 100%); }

    /* Bottom Nav (Google Style) */
    .bottom-nav {
      position: fixed; bottom: 0; left: 0; right: 0;
      background: var(--nav-bg); 
      display: flex; justify-content: space-around; align-items: center;
      padding: 8px 10px calc(8px + var(--safe-bottom)); z-index: 100;
    }
    .nav-item {
      display: flex; flex-direction: column; align-items: center; gap: 4px;
      color: var(--text-secondary); font-size: 11px; font-weight: 500; cursor: pointer; padding: 8px 0; width: 64px;
    }
    .nav-icon-bg { width: 64px; height: 32px; border-radius: 16px; display: flex; align-items: center; justify-content: center; transition: all 0.2s;}
    .nav-item svg { width: 22px; height: 22px; }
    .nav-item.active { color: #fff; }
    .nav-item.active .nav-icon-bg { background: rgba(168, 199, 250, 0.15); }
    .nav-item.active svg { color: #A8C7FA; font-weight: 600;}
    """
    
    # Replace old CSS with new material CSS
    content = re.sub(r'<style>.*?</style>', f'<style>\n{material_css}\n</style>', content, flags=re.DOTALL)

    # --- 2. HTML STRUCTURE REPLACEMENT ---
    # Top Header
    new_header = """
    <div class="header">
      <svg class="back-btn" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 12H5M12 19l-7-7 7-7"/></svg>
      <div class="location" onclick="showScreen('searchScreen')"><span id="mainCity">Jena</span></div>
    </div>
    """
    content = re.sub(r'<div class="header">.*?</div>', new_header, content, flags=re.DOTALL)

    # API Update from 7 to 10 days
    content = content.replace("daily=weather_code,temperature_2m_max,temperature_2m_min,sunrise,sunset,uv_index_max", 
                              "daily=weather_code,temperature_2m_max,temperature_2m_min,sunrise,sunset,uv_index_max,precipitation_probability_max&forecast_days=10")
    content = content.replace("hourly=temperature_2m,weather_code,relative_humidity_2m,wind_speed_10m,pressure_msl",
                              "hourly=temperature_2m,weather_code,relative_humidity_2m,wind_speed_10m,pressure_msl,precipitation_probability")


    # Predict Card to AI Weather Report
    ai_report_js = """
      document.getElementById('aiStatusText').textContent = "AI weather report";
      const tempTrend = predictions[5] > c.temperature_2m ? 'Rise' : 'Drop';
      const diff = Math.abs(predictions[5] - c.temperature_2m).toFixed(1);
      document.getElementById('predText').innerHTML = `
        <div class="ai-summary">Live Edge AI interpretation: Expect a ${tempTrend.toLowerCase()}ing temperature curve over the next 6 hours.</div>
        <ul class="ai-bullets">
          <li>Current Edge model prediction active.</li>
          <li>Temperature shifting by ${diff}°C shortly.</li>
          <li>Inference computed natively on this device.</li>
        </ul>
        <div class="ai-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg> AI-generated</div>
      `;
    """
    html_predict = """
    <div class="predict-card">
      <div class="ai-header">
        <div class="ai-title"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2l3 6 6 3-6 3-3 6-3-6-6-3 6-3z"/></svg> <span id="aiStatusText">AI weather report</span></div>
      </div>
      <div id="predText" style="opacity: 0.6; font-size: 14px;">Initializing Edge Model...</div>
    </div>
    """
    content = re.sub(r'<div class="predict-card">.*?</div>\n    </div>', html_predict, content, flags=re.DOTALL)
    # also patch the inner generation logic
    content = re.sub(r"document\.getElementById\('aiStatusText'\)\.textContent = \"EDGE AI LIVE\";.*?`;", ai_report_js.strip(), content, flags=re.DOTALL)

    # Hourly Forecast formatting
    new_hourly_js = """
        const precip = h.precipitation_probability ? h.precipitation_probability[hourIndex] : 0;
        hourlyHtml += `
          <div class="hourly-item">
            <div class="h-temp">${Math.round(displayTemp)}°</div>
            <div class="h-icon">${info.icon}</div>
            <div class="h-precip">${precip}%</div>
            <div class="h-time">${timeStr}</div>
          </div>`;
    """
    content = re.sub(r"const displayTemp = i === 0 \? c\.temperature_2m : predictions\[i-1\];.*?</div>`;", "const displayTemp = i === 0 ? c.temperature_2m : predictions[i-1];" + new_hourly_js, content, flags=re.DOTALL)
    content = content.replace("<div class=\"card-title\">HOURLY FORECAST</div>", "<div class=\"card-title\"><svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\"><circle cx=\"12\" cy=\"12\" r=\"10\"/><path d=\"M12 6v6l4 2\"/></svg> Hourly forecast</div>")


    # Daily 10-day formatting
    new_daily_js = """
  // 10-Day forecast (Horizontal Pills)
  let dailyHtml = '<div class="daily-scroll">';
  const dayNamesArr = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const limit = d.time.length > 10 ? 10 : d.time.length;
  for (let i = 0; i < limit; i++) {
    const date = new Date(d.time[i]);
    const dayName = i === 0 ? 'Today' : dayNamesArr[date.getDay()];
    const dateFormatted = date.getDate() + '/' + (date.getMonth()+1).toString().padStart(2,'0');
    const info = getWeatherInfo(d.weather_code[i]);
    const lo = Math.round(d.temperature_2m_min[i]);
    const hi = Math.round(d.temperature_2m_max[i]);
    const precip = d.precipitation_probability_max ? d.precipitation_probability_max[i] : (globalMax > 15 ? 10 : 0);
    
    dailyHtml += `
      <div class="daily-pill ${i === 0 ? 'today' : ''}">
        <span class="d-hi">${hi}°</span>
        <span class="d-lo">${lo}°</span>
        <span class="d-icon">${info.icon}</span>
        <span class="d-precip">${precip}%</span>
        <span class="d-day">${dayName}</span>
        <span class="d-date">${dateFormatted}</span>
      </div>`;
  }
  dailyHtml += '</div>';
  document.getElementById('dailyForecast').innerHTML = dailyHtml;
"""
    content = re.sub(r"// 7-Day forecast.*?document\.getElementById\('dailyForecast'\)\.innerHTML = dailyHtml;", new_daily_js.strip(), content, flags=re.DOTALL)
    content = content.replace("<div class=\"card-title\">7-DAY FORECAST</div>", "")
    content = content.replace("<div class=\"card\" id=\"dailyForecast\"></div>", '<div class="card" style="background:transparent; padding:0;"><div class="card-title" style="padding-left:16px;"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg> 10-day forecast</div><div id="dailyForecast"></div></div>')

    # MAP CARD
    map_card = """
    <div class="card map-card">
      <div class="map-title"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg> Weather map</div>
      <img class="map-img" src="https://static-maps.yandex.ru/1.x/?lang=en_US&ll=${c.lon},${c.lat}&z=9&l=map&size=400,160" alt="Map">
    </div>
    """

    # NEW Detail Grid logic focusing on Material Layout
    new_details_js = f"""
  // Material You Shapes Grid
  const feelsLike = Math.round(c.apparent_temperature);
  const humidity = c.relative_humidity_2m;
  const wind = c.wind_speed_10m;
  const pressure = Math.round(c.pressure_msl);
  const visibility = "16"; // mock
  const aqi = 3; // mock
  
  document.getElementById('detailGrid').innerHTML = `
    {map_card}
    <div class="detail-grid">
      <div class="m-card">
        <div class="m-label"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 22l-2-2m4 2l-2-2m4 2l-2-2m4 2l-2-2M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/></svg> Precipitation</div>
        <div class="m-val"><1.3<span class="m-unit"> mm</span></div>
        <div class="m-desc">Total rain for the day</div>
      </div>
      <div class="m-card m-circle">
        <div class="m-label"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2"/></svg> Wind</div>
        <div class="m-val">${{wind}}<span class="m-unit"> km/h</span></div>
      </div>
      <div class="m-card">
        <div class="m-label"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/><circle cx="12" cy="12" r="4"/></svg> Sunrise and sunset</div>
        <svg style="margin-top:20px; width:100%; height:40px;" viewBox="0 0 100 40"><path d="M 0 40 Q 50 -10 100 40" fill="rgba(255,255,255,0.08)" stroke="rgba(255,255,255,0.2)" stroke-width="2"/></svg>
        <div style="font-size:12px; margin-top:8px;">☼ ${{sunrise}}<br>☾ ${{sunset}}</div>
      </div>
      <div class="m-card m-circle shape-scallop">
        <div class="m-label"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42"/></svg> UV index</div>
        <div class="m-val">2</div>
        <div style="color:var(--text-secondary); font-size:12px;">Low</div>
      </div>
      <div class="m-card">
        <div class="m-label"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg> Air quality</div>
        <div class="m-val">${{aqi}}</div>
        <div style="width:100%; height:4px; border-radius:2px; background:linear-gradient(to right, #4CAF50, #FFEB3B, #F44336, #9C27B0); margin:8px 0;"></div>
        <div class="m-desc">Low air pollution</div>
      </div>
      <div class="m-card m-circle">
        <div class="m-label"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg> Visibility</div>
        <div class="m-val">${{visibility}}<span class="m-unit"> km</span></div>
      </div>
      <div class="m-card">
        <div class="shape-wave"></div>
        <div class="m-label" style="margin-top:15px;"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z"/></svg> Humidity</div>
        <div class="m-val">${{humidity}}<span class="m-unit">%</span></div>
        <div class="m-desc" style="display:flex; align-items:center; gap:4px;"><div style="width:24px; height:24px; border-radius:50%; background:rgba(255,255,255,0.1); display:flex; justify-content:center; align-items:center; font-size:10px;">${{feelsLike}}°</div> Dew point</div>
      </div>
      <div class="m-card m-circle">
        <div class="m-label"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/></svg> Pressure</div>
        <div class="m-val">${{pressure}}</div>
        <div class="m-unit">mbar</div>
      </div>
      <div class="m-card" style="grid-column: span 2; display:flex; flex-direction:column; align-items:center;">
        <div class="m-label" style="margin-bottom:12px;"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 21c.5-4.5 2.5-8 7-10 4.5 2 6.5 5.5 7 10"/></svg> Pollen</div>
        <div style="display:flex; width:100%; justify-content:space-around;">
          <div style="text-align:center;"><div style="width:48px;height:48px;border-radius:50%;border:4px solid #4CAF50; border-top-color:rgba(255,255,255,0.1); margin:0 auto 4px;"></div><span style="font-size:12px;">Grass<br>Low</span></div>
          <div style="text-align:center;"><div style="width:48px;height:48px;border-radius:50%;border:4px solid #FF9800; border-right-color:rgba(255,255,255,0.1); margin:0 auto 4px;"></div><span style="font-size:12px;">Tree<br>High</span></div>
          <div style="text-align:center;"><div style="width:48px;height:48px;border-radius:50%;border:4px solid rgba(255,255,255,0.1); margin:0 auto 4px;"></div><span style="font-size:12px;">Weed<br>None</span></div>
        </div>
      </div>
    </div>
  `;
"""
    content = re.sub(r'document\.getElementById\(\'detailGrid\'\)\.innerHTML = `.*?`;', new_details_js.strip(), content, flags=re.DOTALL)

    # Replace Bottom Navbar with updated layout
    new_nav = """
<div class="bottom-nav">
  <div class="nav-item active" onclick="showScreen('weatherScreen')" id="navWeather">
    <div class="nav-icon-bg"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/></svg></div>
    <span>Weather</span>
  </div>
  <div class="nav-item" onclick="showScreen('modelsScreen')" id="navModels">
    <div class="nav-icon-bg"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a4 4 0 0 0-4 4c0 2 2 3 2 6H14c0-3 2-4 2-6a4 4 0 0 0-4-4z"/><rect x="9" y="12" width="6" height="4" rx="1"/><path d="M10 16v1a2 2 0 0 0 4 0v-1"/></svg></div>
    <span>Models</span>
  </div>
  <div class="nav-item" onclick="showScreen('searchScreen')" id="navSearch">
    <div class="nav-icon-bg"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg></div>
    <span>Search</span>
  </div>
  <div class="nav-item" onclick="showScreen('aboutScreen')" id="navAbout">
    <div class="nav-icon-bg"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg></div>
    <span>About</span>
  </div>
</div>
"""
    content = re.sub(r'<div class="bottom-nav">.*?</div>', new_nav.strip(), content, flags=re.DOTALL)


    with open('pwa/index.html', 'w', encoding='utf-8') as f:
        f.write(content)

    print("Material You transformation applied successfully!")

if __name__ == "__main__":
    main()
