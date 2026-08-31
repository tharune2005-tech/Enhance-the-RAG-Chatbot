"""Write a tiny text PDF without extra dependencies (used for the sample handbook)."""

from __future__ import annotations

from pathlib import Path


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def wrap(text: str, width: int = 92) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if len(trial) > width:
            if current:
                lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def write_pdf(path: Path, title: str, paragraphs: list[str]) -> None:
    lines = [title, ""]
    for paragraph in paragraphs:
        lines.extend(wrap(paragraph))
        lines.append("")

    stream_lines = ["BT", "/F1 14 Tf"]
    y = 740
    for i, line in enumerate(lines):
        font = "/F1 14 Tf" if i == 0 else "/F1 11 Tf"
        stream_lines.append(font)
        stream_lines.append(f"1 0 0 1 54 {y} Tm ({_escape(line)}) Tj")
        y -= 18 if i == 0 else 15
        if y < 54:
            break
    stream_lines.append("ET")
    stream = ("\n".join(stream_lines) + "\n").encode("latin-1", "replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode("ascii"))
        out.extend(obj)
        out.extend(b"\nendobj\n")

    xref_at = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    out.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_at}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(bytes(out))


if __name__ == "__main__":
    dest = Path(__file__).resolve().parent.parent / "data" / "sample_intern_handbook.pdf"
    write_pdf(
        dest,
        "UnsaidTalks Intern Handbook 2026",
        [
            "This handbook is a sample document used to test runtime PDF ingestion in the RAG chatbot. It is not an official UnsaidTalks policy.",
            "Intern stipend: INR 15000 per month, paid on the last working day.",
            "Office hours: 10:00 to 18:00 IST, Monday to Friday. Remote interns overlap 11:00 to 16:00 IST.",
            "Referral code for intern office hours is UT-INTERN-2026.",
            "Leave: 1.5 days of casual leave per month. Unused leave does not carry forward.",
            "Laptop policy: interns use a personal machine. UnsaidTalks does not ship hardware for remote internships.",
            "Weekly sync: Fridays at 16:30 IST with the intern manager.",
        ],
    )
    print(f"Wrote {dest}")
