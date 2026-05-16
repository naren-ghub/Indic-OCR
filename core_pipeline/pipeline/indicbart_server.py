import json
import sys
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def run_server():
    try:
        print("[IndicBART] Starting initialization...", file=sys.stderr)
        model_id = "ai4bharat/IndicBART"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[IndicBART] Loading tokenizer ({model_id})...", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(model_id, do_lower_case=False, use_fast=False, keep_accents=True)
        print(f"[IndicBART] Loading model ({device})...", file=sys.stderr)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
        model.eval()
        print("[IndicBART] Model loaded.", file=sys.stderr)

        # Send ready signal
        sys.stdout.buffer.write(json.dumps({"status": "ready"}).encode("utf-8") + b"\n")
        sys.stdout.buffer.flush()

    except Exception as e:
        sys.stdout.buffer.write(json.dumps({"error": str(e)}).encode("utf-8") + b"\n")
        sys.stdout.buffer.flush()
        sys.exit(1)

    # Listen for requests
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            break
        
        try:
            req = json.loads(line.decode("utf-8"))
            text = req.get("text", "")
            lang_code = req.get("lang_code", "ta")
            
            if not text.strip():
                resp = {"corrected": ""}
            else:
                lang_tag = f"<2{lang_code}>"
                input_text = text + " </s> " + lang_tag
                
                inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        use_cache=True,
                        num_beams=4,
                        max_length=256,
                        decoder_start_token_id=tokenizer.convert_tokens_to_ids(lang_tag)
                    )
                    
                corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Manual cleanup of remaining tags
                for tag in [f"<2{lang_code}>", "</s>", "<s>"]:
                    corrected = corrected.replace(tag, "")
                
                resp = {"corrected": corrected.strip()}
                
            sys.stdout.buffer.write(json.dumps(resp, ensure_ascii=False).encode("utf-8") + b"\n")
            sys.stdout.buffer.flush()
            
        except Exception as e:
            err = {"error": str(e)}
            sys.stdout.buffer.write(json.dumps(err).encode("utf-8") + b"\n")
            sys.stdout.buffer.flush()

if __name__ == "__main__":
    run_server()
