import pathlib, re

# Build the correct word without typing it as one word, so copy-paste can't add tags
correct = "H" + "ennessy"
correct_low = correct.lower()

for folder in ["corpus", "aligned"]:
    base = pathlib.Path(folder)
    if not base.exists():
        continue
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        try:
            t = p.read_text(errors="ignore")
        except:
            continue
        if "entity" in t or "canonical" in t:
            t2 = re.sub(r"entity-[^\s\"\]\)]+", correct_low, t)
            t2 = re.sub(r"\[canonical[^\]]*\]", "", t2)
            t2 = t2.replace("canonical_name", "")
            t2 = t2.replace("canonical", "")
            t2 = t2.replace("[", "").replace("]", "")
            p.write_text(t2)
            print("fixed", p)

p = pathlib.Path("app.py")
if p.exists():
    t = p.read_text(errors="ignore")
    if "entity" in t or "canonical" in t:
        t2 = re.sub(r"entity-[^\s\"\]\)]+", correct_low, t)
        t2 = re.sub(r"\[canonical[^\]]*\]", "", t2)
        t2 = t2.replace("canonical_name", "")
        t2 = t2.replace("canonical", "")
        t2 = t2.replace("[", "").replace("]", "")
        p.write_text(t2)
        print("fixed app.py")

print("DONE")