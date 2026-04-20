import re

def main():
    with open('pwa/index.html', 'r', encoding='utf-8') as f:
        content = f.read()

    duplicate_block = """  // Material You Shapes Grid
  const feelsLike = Math.round(c.apparent_temperature);
  const humidity = c.relative_humidity_2m;
  const wind = c.wind_speed_10m;
  const pressure = Math.round(c.pressure_msl);
  const visibility = "16"; // mock
  const aqi = 3; // mock"""
    
    # We remove the duplicate variable declarations. 
    # But wait! We need `visibility` and `aqi` since they were novel.
    fixed_block = """  // Material You Shapes Grid
  const visibility = "16"; // mock
  const aqi = 3; // mock"""

    content = content.replace(duplicate_block, fixed_block)

    with open('pwa/index.html', 'w', encoding='utf-8') as f:
        f.write(content)

    print("Duplicates removed!")

if __name__ == "__main__":
    main()
