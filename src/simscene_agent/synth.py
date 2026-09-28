from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List


def _svg_header(width: int, height: int, title: str) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#eef2f7"/>',
        f'<text x="24" y="34" font-family="sans-serif" font-size="22" fill="#1f2937">{title}</text>',
    ]


def _box(x: float, y: float, w: float, h: float, fill: str, label: str = "") -> List[str]:
    out = [f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" fill="{fill}" stroke="#243447" stroke-width="2"/>']
    if label:
        out.append(f'<text x="{x+w/2:.1f}" y="{y+h/2+5:.1f}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#111827">{label}</text>')
    return out


def _circle(cx: float, cy: float, r: float, fill: str, label: str = "") -> List[str]:
    out = [f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" stroke="#243447" stroke-width="2"/>']
    if label:
        out.append(f'<text x="{cx:.1f}" y="{cy+5:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#111827">{label}</text>')
    return out


def _render_kitchen(frame: int, out: Path) -> None:
    angle = frame * 12
    lines = _svg_header(900, 560, f"kitchen sequence / frame {frame:02d}")
    lines += ['<polygon points="80,470 820,470 700,330 190,330" fill="#d9e1e8" stroke="#64748b"/>', '<rect x="100" y="90" width="700" height="240" fill="#f7f1e8" stroke="#64748b"/>']
    lines += _box(520, 180, 180, 150, "#b98555", "cabinet")
    # door rotates visually around its left hinge
    dx = 610 + 55 * math.sin(math.radians(angle))
    dy = 205 - 35 * math.sin(math.radians(angle))
    lines.append(f'<polygon points="610,205 {dx:.1f},{dy:.1f} {dx+80:.1f},{dy+8:.1f} 690,215" fill="#d0a06d" stroke="#243447" stroke-width="2"/>')
    lines += _box(250, 360, 300, 35, "#8b5e3c", "table")
    for lx in (270, 510):
        lines.append(f'<line x1="{lx}" y1="395" x2="{lx-15}" y2="470" stroke="#5b3b25" stroke-width="12"/>')
    lines += _circle(360, 346, 15, "#60a5fa", "cup")
    lines += _box(120, 390, 70, 20, "#94a3b8", "robot")
    lines += ['<text x="570" y="150" font-family="sans-serif" font-size="14" fill="#334155">door angle: %d°</text>' % min(angle, 90)]
    lines.append('</svg>')
    out.write_text("\n".join(lines), encoding="utf-8")


def _render_office(frame: int, out: Path) -> None:
    lines = _svg_header(900, 560, f"office sequence / frame {frame:02d}")
    lines += ['<rect x="90" y="90" width="720" height="370" fill="#f8fafc" stroke="#64748b"/>']
    lines += _box(210, 300, 390, 34, "#c08457", "desk")
    lines += _box(330, 180, 160, 115, "#94a3b8", "monitor")
    lines += _box(660, 170, 85, 180, "#a78bfa", "shelf")
    for i in range(4):
        lines += _box(675, 195 + i * 37, 55, 22, "#fbbf24", f"obj{i+1}")
    x = 290 + frame * 20
    lines += _circle(x, 368, 16, "#f87171", "robot")
    lines += ['<path d="M290 368 C410 430 580 420 700 360" fill="none" stroke="#ef4444" stroke-width="3" stroke-dasharray="8 6"/>', '<text x="110" y="500" font-family="sans-serif" font-size="15" fill="#475569">reachability path is recorded as a task constraint</text>', '</svg>']
    out.write_text("\n".join(lines), encoding="utf-8")


def _render_shelf(frame: int, out: Path) -> None:
    lines = _svg_header(900, 560, f"shelf sequence / frame {frame:02d}")
    lines += ['<rect x="100" y="90" width="700" height="370" fill="#fff7ed" stroke="#64748b"/>']
    lines += _box(250, 140, 420, 270, "#a16207", "shelf")
    for y in (190, 260, 330):
        lines += _box(270, y, 380, 14, "#fcd34d")
    for i, x in enumerate((300, 380, 460, 540)):
        lines += _box(x, 205 + (frame % 3) * 2, 40, 45, "#60a5fa" if i % 2 else "#fb7185", f"o{i+1}")
    lines += ['<text x="250" y="500" font-family="sans-serif" font-size="15" fill="#475569">support and containment constraints are checked after each frame</text>', '</svg>']
    out.write_text("\n".join(lines), encoding="utf-8")


def generate_sequence(scene_id: str, root: Path, frames: int = 8) -> List[str]:
    renderers = {"kitchen": _render_kitchen, "office": _render_office, "shelf": _render_shelf}
    renderer = renderers[scene_id]
    seq_dir = root / "image_sequence"
    seq_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(frames):
        p = seq_dir / f"frame_{i:03d}.svg"
        renderer(i, p)
        paths.append(str(p))
    return paths
