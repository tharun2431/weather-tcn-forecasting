import re

def main():
    with open('pwa/index.html', 'r', encoding='utf-8') as f:
        content = f.read()

    # Replacements
    replacements = {
        'LOADING AI': 'LOADING MODEL',
        'LIVE AI': 'LIVE MODEL',
        'AI weather report': 'Model prediction',
        'Initializing Edge AI model...': 'Initializing Edge model...',
        'Edge AI inference:': 'Edge model inference:',
        'AI-generated': 'Model-generated'
    }

    for old, new in replacements.items():
        content = content.replace(old, new)

    with open('pwa/index.html', 'w', encoding='utf-8') as f:
        f.write(content)

    print("Replaced AI wording.")

if __name__ == "__main__":
    main()
