import fitz
doc = fitz.open(r"C:\Users\ADMIN\Desktop\New folder\Assessment Brief - Deep Learning Applications - CMP-L016.pdf")
with open("assignment_brief.txt", "w", encoding="utf-8") as f:
    for i, page in enumerate(doc):
        f.write(f"--- PAGE {i+1} ---\n")
        f.write(page.get_text())
        f.write("\n\n")
print(f"Extracted {len(doc)} pages to assignment_brief.txt")
