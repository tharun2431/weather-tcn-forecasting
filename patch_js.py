import re

def main():
    with open('pwa/index.html', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Fix globalMax ReferenceError
    bad_line = "const precip = d.precipitation_probability_max ? d.precipitation_probability_max[i] : (globalMax > 15 ? 10 : 0);"
    good_line = "const precip = d.precipitation_probability_max ? d.precipitation_probability_max[i] : (hi > 15 ? 10 : 0);"
    content = content.replace(bad_line, good_line)

    # 2. Fix aiStatusBadge null pointer exception
    ai_script = """      document.getElementById('aiStatusText').textContent = "AI weather report";
      document.getElementById('aiStatusBadge').classList.add('live');
      document.getElementById('aiStatusBadge').style.borderColor = "var(--accent-green)";"""
      
    fixed_ai = """      document.getElementById('aiStatusText').textContent = "AI weather report";"""
    content = content.replace(ai_script, fixed_ai)

    # Note: fallback replace in case it was somewhat modified
    content = re.sub(r"document\.getElementById\('aiStatusBadge'\).*?;", "", content)

    # 3. Update API to 10 days
    url_bad = "&daily=temperature_2m_max,temperature_2m_min,weather_code,sunrise,sunset,uv_index_max&timezone=auto&past_days=7&forecast_days=7"
    url_good = "&daily=temperature_2m_max,temperature_2m_min,weather_code,sunrise,sunset,uv_index_max,precipitation_probability_max&timezone=auto&past_days=7&forecast_days=10"
    content = content.replace(url_bad, url_good)

    with open('pwa/index.html', 'w', encoding='utf-8') as f:
        f.write(content)

    print("JS Errors patched!")

if __name__ == "__main__":
    main()
