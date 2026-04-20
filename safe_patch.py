import re

def main():
    with open('pwa/index.html', 'r', encoding='utf-8') as f:
        content = f.read()

    # The content is now the clean `0f47db0` version.
    
    # 1. Remove UV Index
    uv_block = r"""    <div class="detail-card">
      <div class="d-label">☀️ UV INDEX</div>
      <div class="d-value" style="color:\$\{uvColor\}">\$\{uvIndex\.toFixed\(0\)\}</div>
      <div class="d-note">\$\{uvLevel\}</div>
    </div>\n"""
    content = re.sub(uv_block, '', content)
    
    uv_block2 = r"""    <div class="detail-card">\s*<div class="d-label">☀️ UV INDEX</div>\s*<div class="d-value" style="color:\$\{uvColor\}">\$\{uvIndex\.toFixed\(0\)\}</div>\s*<div class="d-note">\$\{uvLevel\}</div>\s*</div>\n"""
    content = re.sub(uv_block2, '', content)

    # 2. Upgrade lightning to premium emoji ✨ without corrupting file
    content = content.replace('<div class="predict-icon">⚡</div>', '<div class="predict-icon">✨</div>')

    with open('pwa/index.html', 'w', encoding='utf-8') as f:
        f.write(content)

    print("Patched safely.")

if __name__ == "__main__":
    main()
