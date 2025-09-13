# raisin_chat_cli.py — simple CLI chat; attaches a current screenshot each turn
import os, io, base64
from PIL import Image
import mss
from openai import OpenAI

MODEL        = "gpt-4o-mini"  # vision-capable
MONITOR_IDX  = 0              # 0 = all displays, 1 = primary
JPEG_MAX_W   = 1280           # downscale to save bandwidth
JPEG_QUALITY = 80

SYSTEM = (
  "You can see the user's screen via images captured each turn.\n"
  "Be a gentle, step-by-step guide for older adults.\n"
  "Rules:\n"
  "- One clear instruction at a time, plain words.\n"
  "- Refer to on-screen text/positions (e.g., 'Click Compose on the left').\n"
  "- If the screen doesn't match, ask what the user sees in the top-left.\n"
  "- Never read/store passwords; if a password field is visible, say 'awaiting private input'."
)

def grab_frame_data_url():
    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[MONITOR_IDX])
    img = Image.frombytes("RGB", shot.size, shot.rgb)
    w, h = img.size
    if w > JPEG_MAX_W:
        img = img.resize((JPEG_MAX_W, int(h * JPEG_MAX_W / w)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY first: export OPENAI_API_KEY='YOUR_NEW_KEY'")
    client = OpenAI(api_key=api_key)

    messages = [{"role":"system","content":SYSTEM}]
    print("Raisin chat ready. Type your message. Type 'quit' to exit.")

    while True:
        try:
            user = input("\nYou: ").strip()
            if user.lower() in {"quit","exit","q"}:
                print("Bye!")
                break

            img_url = grab_frame_data_url()
            messages.append({"role":"user","content":[
                {"type":"text","text":user},
                {"type":"image_url","image_url":{"url":img_url}}
            ]})

            resp = client.chat.completions.create(
                model=MODEL,
                temperature=0.2,
                messages=messages
            )
            reply = resp.choices[0].message.content.strip()
            print(f"\nRaisin: {reply}")
            messages.append({"role":"assistant","content":reply})

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print("ERROR:", e)

if __name__ == "__main__":
    main()
