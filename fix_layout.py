import re

def main():
    with open('pwa/index.html', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Map card should span 2 columns
    bad_map = '<div class="card map-card">'
    good_map = '<div class="card map-card" style="grid-column: span 2;">'
    content = content.replace(bad_map, good_map)

    # 2. Prevent nested detail-grid
    # The structure I injected has `<div class="detail-grid">` right after the map-card.
    # And it has an ending `</div>` at the very end of the innerHTML string.
    
    bad_inner_grid = '''    <div class="detail-grid">
      <div class="m-card">
        <div class="m-label"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 22l-2-2m4 2l-2-2m4 2l-2-2m4 2l-2-2M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/></svg> Precipitation</div>'''

    good_inner_grid = '''      <div class="m-card">
        <div class="m-label"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 22l-2-2m4 2l-2-2m4 2l-2-2m4 2l-2-2M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/></svg> Precipitation</div>'''
        
    content = content.replace(bad_inner_grid, good_inner_grid)
    
    # 3. Remove the extra closing div for the nested grid at the end of the template literal
    bad_end_grid = '''        </div>
      </div>
    </div>
  `;

  // Show weather screen'''
  
    good_end_grid = '''        </div>
      </div>
  `;

  // Show weather screen'''

    content = content.replace(bad_end_grid, good_end_grid)

    with open('pwa/index.html', 'w', encoding='utf-8') as f:
        f.write(content)

    print("Layout fixed!")

if __name__ == "__main__":
    main()
