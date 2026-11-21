import re

with open("app.py", "r", encoding="utf-8", errors="ignore") as f:
    t = f.read()

def rep(m):
    s = m.group(0)
    if "canonical_name=" in s:
        v = s.split("canonical_name=")[1].split("]")[0].split("|")[0]
        # Whisper variable must stay lowercase
        if v == "Whis" + "per":
            v = "whis" + "per"
        return v
    return ""

t2 = re.sub(r"\[entity-[^\]]+\]", rep, t)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(t2)

print("Cleaned all [entity] tags. Now check app.py")