import re

def main():
    with open('pwa/index.html', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Add .screen rules back to CSS
    css_injection = """
    /* Navigation Screens */
    .screen { display: none; animation: fadeIn 0.3s ease-out; }
    .screen.active { display: block; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

    /* Main Weather */"""
    
    content = content.replace("    /* Main Weather */", css_injection)

    # 2. Remove the Pollen option
    pollen_block = """      <div class="m-card" style="grid-column: span 2; display:flex; flex-direction:column; align-items:center;">
        <div class="m-label" style="margin-bottom:12px;"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 21c.5-4.5 2.5-8 7-10 4.5 2 6.5 5.5 7 10"/></svg> Pollen</div>
        <div style="display:flex; width:100%; justify-content:space-around;">
          <div style="text-align:center;"><div style="width:48px;height:48px;border-radius:50%;border:4px solid #4CAF50; border-top-color:rgba(255,255,255,0.1); margin:0 auto 4px;"></div><span style="font-size:12px;">Grass<br>Low</span></div>
          <div style="text-align:center;"><div style="width:48px;height:48px;border-radius:50%;border:4px solid #FF9800; border-right-color:rgba(255,255,255,0.1); margin:0 auto 4px;"></div><span style="font-size:12px;">Tree<br>High</span></div>
          <div style="text-align:center;"><div style="width:48px;height:48px;border-radius:50%;border:4px solid rgba(255,255,255,0.1); margin:0 auto 4px;"></div><span style="font-size:12px;">Weed<br>None</span></div>
        </div>
      </div>"""

    content = content.replace(pollen_block, "")

    with open('pwa/index.html', 'w', encoding='utf-8') as f:
        f.write(content)

    print("Screens patch and Pollen removal applied.")

if __name__ == "__main__":
    main()
