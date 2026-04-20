import re

def main():
    with open('pwa/index.html', 'r', encoding='utf-8') as f:
        content = f.read()

    # The actual implementation we need to replace is:
    # function getWeatherInfo(code) {
    #   const map = {
    #     0:  { icon: '☀️', desc: 'Clear Sky' },
    #     ...
    #   };
    #   return map[code] || { icon: '❓', desc: 'Unknown' };
    # }

    new_get_weather = """function getWeatherInfo(code) {
  function getIcon(c) {
    if (c <= 1) return '<svg viewBox="0 0 24 24" fill="none" stroke="#FFD60A" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>';
    if (c <= 3) return '<svg viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/></svg>';
    if (c <= 49) return '<svg viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/><path d="M8 20v2M12 20v2M16 20v2"/></svg>';
    if (c <= 69) return '<svg viewBox="0 0 24 24" fill="none" stroke="#0A84FF" stroke-width="2"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/><path d="M8 22l-2-2M12 22l-2-2M16 22l-2-2"/></svg>';
    if (c <= 79) return '<svg viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/><path d="M8 23l-1-1M12 23l-1-1M16 23l-1-1"/></svg>';
    return '<svg viewBox="0 0 24 24" fill="none" stroke="#BF5AF2" stroke-width="2"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/><path d="M13 22l-3-6 4-2-6-8"/></svg>';
  }
  let desc = 'Clear';
  if (code == 1 || code == 2) desc = 'Partly Cloudy';
  if (code == 3) desc = 'Overcast';
  if (code >= 45 && code <= 48) desc = 'Fog';
  if (code >= 51 && code <= 67) desc = 'Rain';
  if (code >= 71 && code <= 77) desc = 'Snow';
  if (code >= 95) desc = 'Thunderstorm';
  return { icon: getIcon(code), desc };
}"""

    # We will just replace it using regex
    content = re.sub(r'function getWeatherInfo\(code\)\s*\{\s*const map = \{.*?\};\s*return map\[code\].*?\};?\s*\}', new_get_weather, content, flags=re.DOTALL)

    content = content.replace("document.getElementById('mainIcon').textContent = weatherInfo.icon;", "document.getElementById('mainIcon').innerHTML = weatherInfo.icon;")

    with open('pwa/index.html', 'w', encoding='utf-8') as f:
        f.write(content)

    print("Emojis swapped out successfully!")

if __name__ == "__main__":
    main()
