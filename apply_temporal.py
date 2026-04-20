import re

def main():
    with open('pwa/index.html', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Add CSS for the new temporal pills
    temporal_css = """
    /* Temporal Breakdown */
    .d-temporal {
      display: flex; gap: 8px; flex: 1; justify-content: center; margin: 0 10px;
    }
    .t-pill {
      background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
      border-radius: 12px; padding: 4px 8px; display: flex; flex-direction: column; align-items: center;
      width: 48px;
    }
    .t-pill span:first-child { font-size: 10px; color: var(--text-tertiary); font-weight: 600; text-transform: uppercase; }
    .t-pill span:last-child { font-size: 14px; font-weight: 500; color: var(--text-primary); margin-top: 2px;}
    """
    
    # We inject this CSS right before `/* Detail Grid`
    content = content.replace('    /* Detail Grid', temporal_css + '\n    /* Detail Grid')

    # 2. Modify the JS section
    # Old logic:
    # const left = ((d.temperature_2m_min[i] - globalMin) / range) * 100;
    # const width = ((d.temperature_2m_max[i] - d.temperature_2m_min[i]) / range) * 100;
    # dailyHtml += `...
    #   <div class="d-bar-bg"><div class="d-bar-fill" style="left:${left}%;width:${Math.max(width, 8)}%"></div></div>
    
    old_js = """    const left = ((d.temperature_2m_min[i] - globalMin) / range) * 100;
    const width = ((d.temperature_2m_max[i] - d.temperature_2m_min[i]) / range) * 100;
    dailyHtml += `
      <div class="daily-item">
        <div class="d-day ${i === 0 ? 'today' : ''}">${dayName}</div>
        <div class="d-icon">${info.icon}</div>
        <div class="d-bar-bg"><div class="d-bar-fill" style="left:${left}%;width:${Math.max(width, 8)}%"></div></div>
        <div class="d-temps"><span class="d-hi">${hi}°</span><span class="d-lo">${lo}°</span></div>
      </div>`;"""

    new_js = """    // Get temporal hourly datums if available
    let morn = "--", aft = "--", night = "--";
    const dateStr = d.time[i]; // e.g. "2023-10-25"
    if (h && h.time) {
      const idxMorn = h.time.findIndex(t => t.startsWith(dateStr) && t.includes("08:00"));
      const idxAft = h.time.findIndex(t => t.startsWith(dateStr) && t.includes("14:00"));
      const idxNight = h.time.findIndex(t => t.startsWith(dateStr) && t.includes("20:00"));
      if (idxMorn !== -1) morn = Math.round(h.temperature_2m[idxMorn]);
      if (idxAft !== -1) aft = Math.round(h.temperature_2m[idxAft]);
      if (idxNight !== -1) night = Math.round(h.temperature_2m[idxNight]);
    }
    
    dailyHtml += `
      <div class="daily-item">
        <div class="d-day ${i === 0 ? 'today' : ''}">${dayName}</div>
        <div class="d-icon">${info.icon}</div>
        <div class="d-temporal">
          <div class="t-pill"><span>8 AM</span><span>${morn}°</span></div>
          <div class="t-pill"><span>2 PM</span><span>${aft}°</span></div>
          <div class="t-pill"><span>8 PM</span><span>${night}°</span></div>
        </div>
        <div class="d-temps"><span class="d-hi">${hi}°</span><span class="d-lo">${lo}°</span></div>
      </div>`;"""

    content = content.replace(old_js, new_js)

    with open('pwa/index.html', 'w', encoding='utf-8') as f:
        f.write(content)

    print("Replaced 7-Day bar with temporal pills!")

if __name__ == "__main__":
    main()
