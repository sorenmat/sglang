"""Merge cover + body into the final report PDF."""
from pypdf import PdfReader, PdfWriter

A4_W, A4_H = 595.28, 841.89
OUT = "/home/smo/code/sglang-perf/benchmark/context_certificate/report_build"


def normalize(page):
    w, h = float(page.mediabox.width), float(page.mediabox.height)
    if abs(w - A4_W) > 2 or abs(h - A4_H) > 2:
        page.scale(A4_W / w, A4_H / h)
    return page


writer = PdfWriter()
writer.add_page(normalize(PdfReader(f"{OUT}/cover.pdf").pages[0]))
for p in PdfReader(f"{OUT}/body.pdf").pages:
    writer.add_page(normalize(p))
writer.add_metadata({
    "/Title": "The Overflow Shelf - Plain-Language Results Report",
    "/Author": "Z.ai", "/Creator": "Z.ai",
    "/Subject": "How one AI chip handles documents bigger than its own memory",
})
final = f"{OUT}/host-kv-plain-report.pdf"
with open(final, "wb") as f:
    writer.write(f)
print("merged:", final, len(writer.pages), "pages")
