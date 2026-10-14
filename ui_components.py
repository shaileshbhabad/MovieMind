"""
ui_components.py
Custom styling, CSS themes, SVG placeholders, and visual helper components for MovieMind.
"""

import html
import urllib.parse
from typing import Optional, Dict, Any, List

# MovieMind Cinematic Dark Theme Palette
BG_COLOR = "#0B0D14"
CARD_BG = "#151824"
CARD_HOVER_BG = "#1C2030"
ACCENT_PURPLE = "#9B72F2"
ACCENT_CYAN = "#54C8E8"
TEXT_PRIMARY = "#F4F4FA"
TEXT_SECONDARY = "#A5ABC0"
BORDER_COLOR = "rgba(255, 255, 255, 0.08)"
BORDER_ACCENT = "rgba(155, 114, 242, 0.3)"

CUSTOM_CSS = f"""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&family=Space+Grotesk:wght@500;700&display=swap');

/* Global App Styling */
.stApp {{
    background-color: {BG_COLOR};
    color: {TEXT_PRIMARY};
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}}

/* Header and Navigation adjustments */
header[data-testid="stHeader"] {{
    background-color: transparent !important;
}}

/* Main container spacing */
.block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1400px;
}}

/* MovieMind Brand Logo Banner */
.moviemind-logo-container {{
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 10px 0 20px 0;
}}

.moviemind-logo-icon {{
    background: linear-gradient(135deg, {ACCENT_PURPLE} 0%, {ACCENT_CYAN} 100%);
    width: 48px;
    height: 48px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 8px 24px rgba(155, 114, 242, 0.35);
}}

.moviemind-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #FFFFFF 0%, #D8C5FF 50%, {ACCENT_CYAN} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.1;
}}

.moviemind-tagline {{
    color: {TEXT_SECONDARY};
    font-size: 0.95rem;
    margin-top: 2px;
}}

/* Hero Section */
.hero-box {{
    background: linear-gradient(135deg, rgba(25, 28, 41, 0.9) 0%, rgba(20, 23, 34, 0.95) 100%);
    border: 1px solid {BORDER_COLOR};
    border-radius: 20px;
    padding: 32px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
}}

.hero-box::before {{
    content: '';
    position: absolute;
    top: -50px;
    right: -50px;
    width: 250px;
    height: 250px;
    background: radial-gradient(circle, rgba(155, 114, 242, 0.15) 0%, rgba(0, 0, 0, 0) 70%);
    border-radius: 50%;
    pointer-events: none;
}}

.hero-headline {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.3rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 10px;
    line-height: 1.2;
}}

.hero-headline span {{
    background: linear-gradient(90deg, {ACCENT_PURPLE} 0%, {ACCENT_CYAN} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.hero-desc {{
    font-size: 1.05rem;
    color: {TEXT_SECONDARY};
    max-width: 820px;
    line-height: 1.6;
    margin-bottom: 18px;
}}

/* Trending Chips */
.trending-chip-label {{
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: {ACCENT_CYAN};
    font-weight: 600;
    margin-bottom: 8px;
}}

/* Spotlight Card for Selected Movie */
.spotlight-card {{
    background: linear-gradient(135deg, #181B28 0%, #12141F 100%);
    border: 1px solid {BORDER_ACCENT};
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 30px;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.5), 0 0 20px rgba(155, 114, 242, 0.12);
}}

.spotlight-badge {{
    display: inline-block;
    background: rgba(155, 114, 242, 0.2);
    border: 1px solid {ACCENT_PURPLE};
    color: #E2D4FF;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 14px;
}}

.spotlight-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 6px;
}}

.spotlight-meta {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 16px;
    margin-bottom: 14px;
    font-size: 0.95rem;
    color: {TEXT_SECONDARY};
}}

.spotlight-meta-item {{
    display: flex;
    align-items: center;
    gap: 6px;
}}

.spotlight-rating {{
    background: rgba(255, 193, 7, 0.15);
    border: 1px solid rgba(255, 193, 7, 0.4);
    color: #FFD54F;
    padding: 3px 10px;
    border-radius: 8px;
    font-weight: 700;
}}

.spotlight-tagline {{
    font-style: italic;
    color: #D3C4F5;
    margin-bottom: 14px;
    font-size: 1rem;
}}

.spotlight-overview {{
    color: #D3D7E8;
    line-height: 1.65;
    font-size: 0.98rem;
    margin-bottom: 18px;
}}

.spotlight-credits {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid {BORDER_COLOR};
    border-radius: 12px;
    padding: 14px 18px;
}}

.credit-title {{
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: {TEXT_SECONDARY};
    margin-bottom: 2px;
}}

.credit-value {{
    font-size: 0.92rem;
    font-weight: 600;
    color: #FFFFFF;
}}

/* Badge Pills */
.genre-pill {{
    display: inline-block;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: #E0E2EC;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 500;
    margin-right: 6px;
    margin-bottom: 6px;
}}

.similarity-badge {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: linear-gradient(135deg, rgba(155, 114, 242, 0.25) 0%, rgba(84, 200, 232, 0.25) 100%);
    border: 1px solid {ACCENT_PURPLE};
    color: #FFFFFF;
    font-size: 0.85rem;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(155, 114, 242, 0.2);
}}

/* Movie Recommendation Card */
.movie-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 16px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    height: 100%;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    margin-bottom: 16px;
}}

.movie-card:hover {{
    transform: translateY(-5px);
    border-color: {ACCENT_PURPLE};
    box-shadow: 0 16px 36px rgba(155, 114, 242, 0.22);
}}

.movie-poster-wrap {{
    position: relative;
    width: 100%;
    aspect-ratio: 2 / 3;
    overflow: hidden;
    background: #0E101A;
}}

.movie-poster-img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 0.35s ease;
}}

.movie-card:hover .movie-poster-img {{
    transform: scale(1.04);
}}

.card-similarity-overlay {{
    position: absolute;
    top: 10px;
    right: 10px;
    z-index: 2;
}}

.movie-card-content {{
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    flex-grow: 1;
}}

.movie-card-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 4px;
    line-height: 1.3;
}}

.movie-card-year {{
    font-size: 0.85rem;
    color: {TEXT_SECONDARY};
    margin-bottom: 8px;
}}

.movie-card-genres {{
    margin-bottom: 10px;
}}

.movie-card-rating {{
    font-size: 0.85rem;
    color: #FFD54F;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 4px;
}}

/* Streamlit Widget Styling Overrides */
div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label {{
    color: {TEXT_PRIMARY} !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
}}

div[data-testid="stSelectbox"] > div {{
    background-color: {CARD_BG} !important;
    border: 1px solid {BORDER_COLOR} !important;
    border-radius: 12px !important;
    color: {TEXT_PRIMARY} !important;
}}

.stButton > button {{
    background: linear-gradient(135deg, {ACCENT_PURPLE} 0%, #7E4AE8 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 16px rgba(155, 114, 242, 0.3) !important;
}}

.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px rgba(155, 114, 242, 0.45) !important;
}}

/* Secondary Buttons */
.stButton > button[kind="secondary"] {{
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid {BORDER_COLOR} !important;
    color: {TEXT_PRIMARY} !important;
    box-shadow: none !important;
}}

.stButton > button[kind="secondary"]:hover {{
    background: rgba(255, 255, 255, 0.1) !important;
    border-color: {ACCENT_PURPLE} !important;
}}

/* Metric Cards */
.metric-box {{
    background: {CARD_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 16px;
    padding: 18px 20px;
    text-align: center;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}}

.metric-num {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    background: linear-gradient(90deg, {ACCENT_PURPLE} 0%, {ACCENT_CYAN} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2px;
}}

.metric-label {{
    color: {TEXT_SECONDARY};
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    background-color: rgba(255, 255, 255, 0.03);
    border-radius: 14px;
    padding: 6px;
    border: 1px solid {BORDER_COLOR};
}}

.stTabs [data-baseweb="tab"] {{
    border-radius: 10px;
    color: {TEXT_SECONDARY};
    font-weight: 600;
    padding: 8px 18px;
    border: none !important;
    background-color: transparent;
}}

.stTabs [aria-selected="true"] {{
    background-color: {ACCENT_PURPLE} !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(155, 114, 242, 0.35);
}}

/* Navigation Radiogroup / Bar */
div[data-testid="stRadio"] > div[role="radiogroup"] {{
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    background-color: rgba(255, 255, 255, 0.03) !important;
    border-radius: 14px !important;
    padding: 6px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    margin-bottom: 24px !important;
}}

div[data-testid="stRadio"] label {{
    flex: 1 1 200px !important;
    text-align: center !important;
    justify-content: center !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    background: transparent !important;
    cursor: pointer !important;
    font-weight: 600 !important;
    color: #A5ABC0 !important;
    transition: all 0.2s ease !important;
    border: 1px solid transparent !important;
}}

div[data-testid="stRadio"] label:hover {{
    color: #FFFFFF !important;
    background: rgba(255, 255, 255, 0.06) !important;
}}

div[data-testid="stRadio"] label[data-checked="true"],
div[data-testid="stRadio"] label:has(input:checked) {{
    background: linear-gradient(135deg, {ACCENT_PURPLE} 0%, #7E4AE8 100%) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(155, 114, 242, 0.35) !important;
    border-color: rgba(155, 114, 242, 0.5) !important;
}}

div[data-testid="stRadio"] label span:first-child {{
    display: none !important;
}}

/* Streamlit Image Styling for Movie Posters */
div[data-testid="stImage"] {{
    border-radius: 14px !important;
    overflow: hidden !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45) !important;
    aspect-ratio: 2 / 3 !important;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease !important;
    background-color: {CARD_BG} !important;
    border: 1px solid {BORDER_COLOR} !important;
}}

div[data-testid="stImage"]:hover {{
    transform: translateY(-4px) !important;
    box-shadow: 0 16px 36px rgba(155, 114, 242, 0.35) !important;
    border-color: {ACCENT_PURPLE} !important;
}}

div[data-testid="stImage"] img {{
    border-radius: 13px !important;
    object-fit: cover !important;
    width: 100% !important;
    height: 100% !important;
    aspect-ratio: 2 / 3 !important;
    display: block !important;
}}

/* Movie Card Meta Info Container */
.movie-meta-box {{
    background: {CARD_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 14px;
    padding: 12px 14px;
    margin-top: 8px;
    margin-bottom: 8px;
    min-height: 130px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
}}

.movie-meta-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.02rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.25;
    margin-bottom: 4px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}}

.movie-meta-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
    font-size: 0.85rem;
}}

.movie-meta-year {{
    color: {TEXT_SECONDARY};
}}

.movie-meta-rating {{
    color: #FFD54F;
    font-weight: 600;
}}

/* Footer */
.moviemind-footer {{
    text-align: center;
    padding: 40px 0 20px 0;
    color: #A5ABC0;
    font-size: 0.85rem;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    margin-top: 50px;
}}

.moviemind-footer a {{
    color: #54C8E8;
    text-decoration: none;
}}
</style>
"""


def render_svg_placeholder(title: str, year: str = "") -> str:
    """Generate a modern, self-contained SVG base64 data URI for movies missing posters."""
    import base64
    clean_title = html.escape(title[:26] + ("..." if len(title) > 26 else ""))
    clean_year = html.escape(str(year)) if year and str(year) != "N/A" else ""

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 750" width="100%" height="100%">
      <defs>
        <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#191C29" />
          <stop offset="50%" stop-color="#141722" />
          <stop offset="100%" stop-color="#0E101A" />
        </linearGradient>
        <linearGradient id="accent" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#9B72F2" />
          <stop offset="100%" stop-color="#54C8E8" />
        </linearGradient>
        <radialGradient id="glow" cx="50%" cy="35%" r="45%">
          <stop offset="0%" stop-color="#9B72F2" stop-opacity="0.25" />
          <stop offset="100%" stop-color="#9B72F2" stop-opacity="0" />
        </radialGradient>
      </defs>
      <rect width="500" height="750" fill="url(#bg)" />
      <circle cx="250" cy="270" r="160" fill="url(#glow)" />
      <g transform="translate(190, 200)" fill="none" stroke="url(#accent)" stroke-width="5" stroke-linecap="round" stroke-linejoin="round">
        <rect x="0" y="20" width="120" height="90" rx="14" />
        <path d="M 0 50 L 120 50" />
        <path d="M 30 20 L 20 50" />
        <path d="M 60 20 L 50 50" />
        <path d="M 90 20 L 80 50" />
        <circle cx="60" cy="72" r="10" fill="#9B72F2" stroke="none" />
      </g>
      <text x="250" y="440" font-family="sans-serif" font-weight="700" font-size="28" fill="#F4F4FA" text-anchor="middle">{clean_title}</text>
      <text x="250" y="485" font-family="sans-serif" font-weight="500" font-size="20" fill="#A5ABC0" text-anchor="middle">{clean_year}</text>
      <rect x="180" y="525" width="140" height="32" rx="16" fill="rgba(155, 114, 242, 0.15)" stroke="rgba(155, 114, 242, 0.4)" stroke-width="1.5" />
      <text x="250" y="546" font-family="sans-serif" font-weight="700" font-size="13" fill="#9B72F2" text-anchor="middle" letter-spacing="1.5">MOVIEMIND</text>
    </svg>"""

    b64_str = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64_str}"


def render_movie_card_html(
    movie: Dict[str, Any],
    similarity_percentage: Optional[str] = None
) -> str:
    """Renders HTML for a cinematic movie card."""
    title = html.escape(movie["title"])
    year = html.escape(str(movie.get("release_year", "N/A")))
    rating = movie.get("vote_average", 0.0)

    placeholder_src = render_svg_placeholder(movie["title"], year)
    poster_src = movie.get("poster_url")
    if not poster_src or not str(poster_src).strip():
        poster_src = placeholder_src

    # Render top 2 genre pills
    genres = movie.get("genres", [])
    genre_pills_html = "".join(
        [f'<span class="genre-pill">{html.escape(g)}</span>' for g in genres[:2]]
    )

    similarity_overlay = ""
    if similarity_percentage:
        similarity_overlay = f"""
        <div class="card-similarity-overlay">
            <span class="similarity-badge">★ {similarity_percentage} Match</span>
        </div>
        """

    card_html = f"""
    <div class="movie-card">
        <div class="movie-poster-wrap">
            {similarity_overlay}
            <img class="movie-poster-img" src="{poster_src}" alt="{title}" loading="lazy" onerror="this.onerror=null;this.src='{placeholder_src}';" />
        </div>
        <div class="movie-card-content">
            <div class="movie-card-title">{title}</div>
            <div class="movie-card-year">{year}</div>
            <div class="movie-card-genres">{genre_pills_html}</div>
            <div class="movie-card-rating">★ {rating}/10</div>
        </div>
    </div>
    """
    return card_html


def render_movie_card_info(
    movie: Dict[str, Any],
    similarity_percentage: Optional[str] = None
) -> str:
    """Renders the HTML metadata box below a movie poster."""
    title = html.escape(movie["title"])
    year = html.escape(str(movie.get("release_year", "N/A")))
    rating = movie.get("vote_average", 0.0)

    # Render top 2 genre pills
    genres = movie.get("genres", [])
    genre_pills_html = "".join(
        [f'<span class="genre-pill">{html.escape(g)}</span>' for g in genres[:2]]
    )

    similarity_html = (
        f'<div style="margin-bottom: 6px;"><span class="similarity-badge">★ {similarity_percentage} Match</span></div>'
        if similarity_percentage else ""
    )

    return (
        f'<div class="movie-meta-box">'
        f'<div>'
        f'{similarity_html}'
        f'<div class="movie-meta-title" title="{title}">{title}</div>'
        f'<div class="movie-meta-row">'
        f'<span class="movie-meta-year">📅 {year}</span>'
        f'<span class="movie-meta-rating">★ {rating}/10</span>'
        f'</div>'
        f'</div>'
        f'<div style="margin-top: 4px;">{genre_pills_html}</div>'
        f'</div>'
    )

