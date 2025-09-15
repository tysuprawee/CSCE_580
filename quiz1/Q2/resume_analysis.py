from pathlib import Path
import re
from collections import Counter

try:
    from PyPDF2 import PdfReader  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None
    print("PyPDF2 not installed; class resume parsing will be skipped.")

DATA_DIR = Path(__file__).parent / 'data'
FIGURES_DIR = Path(__file__).parent / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

STOP_WORDS = set((DATA_DIR / 'stop_words.txt').read_text(encoding='utf-8').split())


def save_svg_bar(data, filename, title):
    width, height = 800, 400
    bar_width = width / len(data)
    max_count = max(c for _, c in data) or 1
    scale = (height - 50) / max_count
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    parts.append(f'<text x="{width/2}" y="20" text-anchor="middle" font-size="16" fill="white">{title}</text>')
    for i, (word, count) in enumerate(data):
        x = i * bar_width
        bar_h = count * scale
        y = height - bar_h - 20
        parts.append(f'<rect x="{x}" y="{y}" width="{bar_width-2}" height="{bar_h}" fill="steelblue"/>')
        parts.append(f'<text x="{x + bar_width/2}" y="{height - 5}" text-anchor="middle" font-size="10" transform="rotate(45 {x + bar_width/2},{height - 5})" fill="white">{word}</text>')
    parts.append('</svg>')
    Path(filename).write_text(''.join(parts), encoding='utf-8')


def tokenize(text: str):
    return re.findall(r'\b\w+\b', text.lower())


# My resume analysis
my_resume_text = (DATA_DIR / 'suprawee_resume.txt').read_text(encoding='utf-8')
my_words = tokenize(my_resume_text)
my_counts = Counter(my_words)
save_svg_bar(my_counts.most_common(20), FIGURES_DIR / 'resume_words.svg', 'Top 20 Resume Words')

my_specific_words = [w for w in my_words if w not in STOP_WORDS]
my_specific_counts = Counter(my_specific_words)
save_svg_bar(my_specific_counts.most_common(20), FIGURES_DIR / 'specific_words.svg', 'Top 20 Specific Words')

# Class resume analysis
class_text = ''
resumes_dir = DATA_DIR / 'resumes'

if PdfReader is not None:
    all_texts = []
    for pdf_path in sorted(resumes_dir.glob('*.pdf')):
        try:
            reader = PdfReader(str(pdf_path))
            text = ''.join(page.extract_text() or '' for page in reader.pages)
            all_texts.append(text)
        except Exception as e:
            print(f'Failed to read {pdf_path}: {e}')
    class_text = '\n'.join(all_texts)
    (DATA_DIR / 'resumes_text.txt').write_text(class_text, encoding='utf-8')
elif (DATA_DIR / 'resumes_text.txt').exists():
    class_text = (DATA_DIR / 'resumes_text.txt').read_text(encoding='utf-8')

if class_text:
    class_words = tokenize(class_text)
    class_counts = Counter(class_words)
    save_svg_bar(class_counts.most_common(20), FIGURES_DIR / 'class_resume_words.svg', 'Top 20 Resume Words (Class)')

    class_specific_words = [w for w in class_words if w not in STOP_WORDS]
    class_specific_counts = Counter(class_specific_words)
    save_svg_bar(class_specific_counts.most_common(20), FIGURES_DIR / 'class_specific_words.svg', 'Top 20 Specific Words (Class)')
else:
    class_specific_words = []
    print("No class resume text available; skipping class analysis.")

# Unique words from my resume
unique_words = sorted(set(my_specific_words) - set(class_specific_words))
(DATA_DIR / 'unique_words.txt').write_text('\n'.join(unique_words), encoding='utf-8')
print(unique_words[:20])
