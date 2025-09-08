CUSTOM_CSS = """
<style>
/* Typography */
.heading {
    text-align: center;
    background: linear-gradient(45deg, #FF4B2B, #FF416C);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subheading {
    text-align: center;
    font-weight: 600;
}

/* Links */
a {
    text-decoration: underline;
    color: #3494E6;
    transition: color 0.3s ease;
}

a:hover {
    color: #FF416C;
}
</style>
"""
TITLE_SUBTITLE_CSS = """
<style>
:root{
/* Brand Colors (screen approximation) */
--brand-blue:  #418FDE;  /* PANTONE 279C */
--brand-brown: #603314;  /* PANTONE 732C */

/* Neutrals */
--title: #221915;
--muted: #5A5A5A;
--panel: #FFFFFF;

/* Effects */
--halo-1: 0 8px 28px rgba(65,143,222,0.18);
--halo-2: 0 0 60px  rgba(65,143,222,0.16);
--border: 1px solid rgba(96,51,20,0.16);

/* Underline thickness */
--u: 12px;
}
@media (prefers-color-scheme: dark){
:root{
    --title:#F3F3F3;
    --muted:#CFCFCF;
    --panel:#111111;
    --border:1px solid rgba(255,255,255,0.10);
    --halo-1: 0 8px 28px rgba(65,143,222,0.30);
    --halo-2: 0 0 80px  rgba(65,143,222,0.28);
}
}

/* Hero */
.hero{
position: relative;
background: var(--panel);
border: var(--border);
border-radius: 18px;
padding: 28px 28px 22px;
margin: 8px 0 18px 0;
box-shadow: var(--halo-1), var(--halo-2);
overflow: hidden;

/* 가운데 정렬 */
text-align: center;
}

/* 상단에서 은은하게 비치는 블루 후광 */
.hero::after{
content:"";
position:absolute; inset:-2px;
pointer-events:none;
border-radius: inherit;
background:
    radial-gradient(1200px 360px at 20% -10%,
    rgba(65,143,222,0.18), transparent 60%);
}

/* Title */
.hero h1{
color: var(--title);
font-size: clamp(28px, 4.2vw, 40px);
line-height: 1.15;
letter-spacing: -0.02em;
margin: 0 0 10px 0;
text-shadow:
    0 2px 12px rgba(65,143,222,0.22),
    0 0 36px rgba(65,143,222,0.16);
}

/* === 강조 밑줄: 브라운→블루 그라디언트 === */
.hero h1 .accent{
/* 두 색이 함께 '밑줄'처럼 보이도록, 텍스트 아래에 gradient 배치 */
background-image: linear-gradient(90deg, var(--brand-brown) 0%, var(--brand-blue) 100%);
background-size: 100% var(--u);
background-position: left calc(100% - 0px);
background-repeat: no-repeat;

/* 줄바꿈 시 각 줄에 동일한 밑줄이 적용되도록 */
-webkit-box-decoration-break: clone;
box-decoration-break: clone;

padding: 0 3px; /* 밑줄 양끝 여백 */
border-radius: 3px; /* 살짝 둥근 밑줄 느낌 */
}

/* Subtitle */
.hero p{
margin: 0;
font-size: clamp(14px, 1.8vw, 16px);
color: var(--muted);
}

/* 기능 칩이 있다면 중앙 정렬 유지 */
.hero .chips{
margin-top: 14px;
display: flex; gap: 8px; flex-wrap: wrap;
justify-content: center;
}
.hero .chip{
font-size: 12px;
padding: 6px 10px;
border-radius: 999px;
border: 1px solid rgba(96,51,20,0.20);
background: rgba(65,143,222,0.10);
color: var(--title);
}

/* CTA 버튼(선택) */
.hero .cta{
display:inline-block; margin-top:16px;
padding:10px 14px; border-radius:999px;
text-decoration:none; color:#fff; font-weight:600;
background: linear-gradient(90deg, var(--brand-brown), var(--brand-blue));
border: 1px solid rgba(96,51,20,0.15);
}
.hero .cta:hover{ filter:brightness(0.96); }
</style>
"""