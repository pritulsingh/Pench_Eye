import io, json, urllib.request, os
from PIL import Image, ImageDraw

def human():
    img = Image.new("RGB", (320, 240), (40, 40, 40))
    d = ImageDraw.Draw(img)
    d.ellipse([130, 30, 190, 90], fill=(220, 180, 140))
    d.rectangle([120, 90, 200, 200], fill=(30, 60, 140))
    d.text((4, 4), os.urandom(4).hex(), fill=(255, 255, 0))
    b = io.BytesIO()
    img.save(b, format="JPEG")
    return b.getvalue()

def blank():
    img = Image.new("RGB", (320, 240), (0, 0, 0))
    px = img.load()
    s = os.urandom(2)
    px[0, 0] = (s[0] % 3, s[1] % 3, 0)
    b = io.BytesIO()
    img.save(b, format="JPEG")
    return b.getvalue()

def post_multipart(url, field, filename, data, content_type="image/jpeg"):
    boundary = "----bound" + os.urandom(8).hex()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req) as r:
        return r.status, json.loads(r.read().decode())

print("HEALTH", urllib.request.urlopen("http://127.0.0.1:8000/health").read().decode())
status = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/api/v1/ml/pipeline/status").read().decode())
print("PIPELINE", json.dumps({
    "ml_mode": status.get("ml_mode"),
    "is_demo": status.get("is_demo"),
    "detector_available": status.get("detector_available"),
    "detector": status.get("detector"),
}, indent=2))
for name, data in [("human.jpg", human()), ("blank.jpg", blank())]:
    code, body = post_multipart("http://127.0.0.1:8000/api/v1/images/upload", "file", name, data)
    print(name, code, json.dumps({k: body.get(k) for k in [
        "status", "tiger_code", "observation_id", "megadescriptor_ran",
        "triage_reason", "reason", "species"
    ]}, indent=2))
    code, diag = post_multipart("http://127.0.0.1:8000/api/v1/ml/pipeline/diagnose", "file", name, data)
    print("diagnose", name, json.dumps({
        "final": diag.get("final"),
        "tiger_detection": diag.get("tiger_detection"),
        "megadescriptor": diag.get("megadescriptor"),
        "triage": diag.get("triage"),
        "detection": diag.get("detection"),
    }, indent=2))
