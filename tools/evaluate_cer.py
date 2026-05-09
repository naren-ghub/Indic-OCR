import sys
import re
import jiwer
from pathlib import Path

# Fix stdout encoding for Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def evaluate_file(file_path: Path, story_name: str, gt_dir: Path):
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Remove header
    header_pattern = r"^Tamil OCR Output — .*\nPages processed: \d+\n={60}\n\n"
    content = re.sub(header_pattern, "", content)
    
    # Remove page separators
    page_sep_pattern = r"={60}\nPAGE \d+\n={60}\n"
    content = re.sub(page_sep_pattern, " ", content)
    
    # Normalize pred text
    pred_text = ' '.join(content.split())
    
    # Load GT
    story_gt_dir = gt_dir / story_name
    if not story_gt_dir.exists():
        print(f"Ground truth directory not found for {story_name}: {story_gt_dir}")
        return
        
    gt_pages = sorted(list(story_gt_dir.glob("page_*.txt")))
    full_gt = []
    for p in gt_pages:
        with open(p, "r", encoding="utf-8") as f:
            full_gt.append(f.read().strip())
            
    gt_text = ' '.join(" ".join(full_gt).split())
    
    cer = jiwer.cer(gt_text, pred_text)
    wer = jiwer.wer(gt_text, pred_text)
    
    print(f"{'='*60}")
    print(f"EVALUATION: {file_path.name}")
    print(f"Ground truth story: {story_name}")
    print(f"{'='*60}")
    print(f"CER: {cer:.2%}")
    print(f"WER: {wer:.2%}")
    print(f"GT Chars: {len(gt_text)}")
    print(f"Pred Chars: {len(pred_text)}")
    print()

if __name__ == "__main__":
    ROOT = Path(__file__).parent
    gt_dir = ROOT / "evaluation" / "ground_truth"
    
    files_to_eval = [
        ("விண்ணபம்_ocr_output.txt", "விண்ணப்பம்"), # Map to correct GT folder name
        ("கண்ணாடி_ocr_output.txt", "கண்ணாடி"),
        ("தொப்பி_ocr_output.txt", "தொப்பி")
    ]
    
    for filename, story in files_to_eval:
        path = ROOT / filename
        evaluate_file(path, story, gt_dir)
