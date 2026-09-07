# PHASE 1: AUDIT REPORT
**Date:** 2026-09-07 | **Status:** Complete — awaiting approval to proceed to Phase 2

---

## 1. FRAMEWORK & BUILD TOOLING

| Aspect | Current State |
|--------|---------------|
| **Framework** | None — pure static HTML + Tailwind CSS (CDN) |
| **Build tool** | None (no package.json, no npm, no build step) |
| **JS framework** | None visible in existing code |
| **Styling** | Tailwind via CDN + 60-line local styles.css |
| **Dependencies** | External: Tailwind CDN, Markmap library (CDN), Unsplash images (CDN) |
| **Package.json** | Does not exist |

**Deployment Model:** Direct push to GitHub Pages from main branch; GitHub auto-deploys static assets.

---

## 2. DIRECTORY STRUCTURE & ROUTING

```
byepesg.github.io/
├── index.html              (Main hero; animated BG; Tailwind CDN)
├── index2.html             (Alt layout; component placeholders; unused?)
├── index3.html             (Another variant; unused?)
├── README.md               (Empty except test comment)
├── styles.css              (60 lines; basic header/nav/grid styles)
├── .gitignore              (Only ignores non-existent "Makefile")
│
├── sections/               (13 HTML fragments)
│   ├── header.html
│   ├── teaching.html       (3 hardcoded courses + Markmap)
│   ├── projects.html       (2 hardcoded projects + images)
│   ├── publications.html   (1 real + 2 placeholder articles)
│   ├── about.html
│   ├── education.html
│   ├── experience.html
│   ├── skills.html
│   ├── honors.html
│   ├── certifications.html
│   ├── contact.html
│   ├── volunteer.html
│   └── roadmap.html
│
├── Projects/
│   └── Images/             (2 GIF files for projects)
│       ├── Basil1.gif
│       ├── Basil2.gif
│       ├── Mercadoyepes1.gif
│       └── Mercadoyepes2.gif
│
└── Roadmaps/
    └── AI_Roadmap.mm.md    (Markmap visualization; Spanish)
```

**Routing Strategy:** Currently unclear. index.html is the live page; index2/3 are orphans or experiments. No visible JS loading sections dynamically. Sections may be manual HTML includes or a Jekyll-like processor expected but missing.

---

## 3. STYLING SYSTEM

- **Color Scheme (observed):**
  - Dark background: `#2f2f2f`, `#333`
  - Text: white, grays
  - Accent: Tailwind defaults (no explicit branding color found)
  - Cards: `#gray-300`, `#gray-200`
  
- **Typography:** Roboto (from styles.css), generic sans-serif fallback
- **Layout:** Grid-based cards (`.projects { display: grid; ... }`) for projects; Flexbox for header nav
- **Responsive:** Basic `meta viewport` tag; no explicit breakpoint testing found in styles
- **State:** Minimal (`:hover` underline on nav links)

**Assessment:** Minimal, ad-hoc styling. No design tokens, no consistent spacing scale, no documented color palette.

---

## 4. EXISTING ACADEMIC/PROFESSIONAL CONTENT

### 4.1 Teaching (Hardcoded in sections/teaching.html)

| Course | Semester | Status | Content |
|--------|----------|--------|---------|
| Programming-III | 2025-I | Stub | No repositories linked; "Activities" placeholder |
| Databases-III | 2025-I | Stub | No repositories linked; "Activities" placeholder |
| Data Structures | 2025-I | Stub | No repositories linked; "Activities" placeholder |

- **Teaching Plan:** Embedded Markmap visualization (Spanish: "Plan de Enseñanza...") with topics (Fundamentos, Herramientas, etc.)
- **Problem:** No connection between hardcoded course cards and Markmap; no metadata (active flag, semester grouping, featured flag)

### 4.2 Projects (Hardcoded in sections/projects.html)

| Project | Stack | Client/Org | Artifacts |
|---------|-------|-----------|-----------|
| Project 1 | VueJS + Laravel | Basil S.A.S. | 2 GIFs |
| Project 2 | ReactJS + PHP | Mercadoyepes | 2 GIFs |

- **Problem:** No metadata (year, featured, categories, role, links to GitHub/demo/docs), only hardcoded HTML card + images

### 4.3 Publications (Mixed in sections/publications.html)

| Title | Type | Status | Details |
|-------|------|--------|---------|
| Research at Purdue University | Real | Active | 9-month internship; DPCTE (Differential Privacy + compression); Advisors: Dr. Jeremiah Blocki, Dr. Seunghoon Lee; STOC submission mentioned |
| Publication Title 1 | Template | Placeholder | `Author A, Author B`; generic abstract |
| Publication Title 2 | Template | Placeholder | `Author A, Author B`; generic abstract |

- **Real content:** Purdue research is present; templates are scaffolds needing replacement
- **Missing:** Links (paper, arXiv, DOI, code), publication year/venue, co-author names

### 4.4 Other Sections (Exist; Content Unclear)

- **about.html, education.html, experience.html, skills.html, honors.html, certifications.html, contact.html, volunteer.html**
  - All exist as fragment files
  - Inspection shows mostly empty or placeholder text (need detailed review)
  - No structured metadata visible

---

## 5. GITHUB PAGES CONFIGURATION

| Setting | Status |
|---------|--------|
| **Workflows (.github/workflows/)** | None — no CI/CD, no build automation |
| **CNAME** | Not present — no custom domain configured |
| **_config.yml** | Not present — not using Jekyll |
| **Base path** | Root (`/`) — assuming root domain deployment |
| **Remote URL** | `https://github.com/byepesg/byepesg.github.io.git` |
| **Active branch** | `main` (presumed deployment source) |
| **Deployment method** | Manual `git push` → GitHub Pages auto-serve |

**Assessment:** No automation, no validation, no build step. Pushing broken HTML directly to live site is possible.

---

## 6. TECHNICAL DEBT & FRAGILITY

| Issue | Severity | Impact |
|-------|----------|--------|
| Content hardcoded in HTML | HIGH | Adding a course/project/publication requires HTML editing; no schema; prone to typos |
| Multiple unused index files (index2, index3) | MEDIUM | Technical debt; unclear intent; maintenance confusion |
| Image paths `/../Projects/Images/` | MEDIUM | Fragile relative paths; may break if structure changes |
| External CDN dependencies | MEDIUM | Tailwind, Markmap, Unsplash require internet; no offline build |
| No content schema or validation | HIGH | Missing required fields go unnoticed until site renders incorrectly |
| No publish/draft workflow | MEDIUM | All content is public by default; no content staging |
| No metadata abstraction | HIGH | Can't independently set `active`, `featured`, `published` without HTML editing |
| No routing abstraction | HIGH | Course/project URLs are implicit (no explicit route definitions) |
| No analytics/SEO config | LOW | No sitemap, robots.txt, or metadata optimization |

---

## 7. RECOMMENDATIONS

### 7.1 PRESERVE

✅ **Existing projects and their images** — They represent real client work; refactor into structured content only
✅ **Purdue research content** — Real, valuable, should be formalized into publications collection
✅ **Teaching courses** — Keep all course names and metadata; migrate into structured format
✅ **Home domain deployment** — GitHub Pages root deployment works; don't introduce custom domain unless required
✅ **Accessibility of current site** — Site is live and working; don't break it during refactor

### 7.2 REFACTOR

🔄 **Hardcoded courses → Structured markdown/YAML with metadata (active, featured, semester)**
🔄 **Hardcoded projects → Structured content collection (year, status, categories, links, role)**
🔄 **Hardcoded publications → Structured metadata (authors, venue, year, links, type)**
🔄 **Teaching plan Markmap → Generate from course/lecture metadata instead of embedding**
🔄 **Image assets → Move to `/public/images/projects/` with optimized formats; verify all paths**
🔄 **styles.css → Migrate to design tokens (spacing, colors, typography) using Tailwind config + CSS variables**
🔄 **sections/*.html → Convert into reusable layout components (not fragments)**

### 7.3 REMOVE

❌ **index2.html and index3.html** — Dead variants; no evidence of active use; causes confusion
❌ **.gitignore "Makefile" entry** — Non-existent file; clean up
❌ **Template publication stubs** (Publication Title 1 & 2) — Placeholder content; replace with real data only
❌ **External Unsplash image URL in hero** — High-bandwidth, non-essential; replace with static background or remove

### 7.4 MIGRATE INTO STRUCTURED CONTENT

📦 **Courses & Teaching**
```
content/courses/
├── 2025-I/
│   ├── programming-iii.md     (metadata: code, title, active, featured, semester, repo, syllabus)
│   ├── databases-iii.md
│   └── data-structures.md
├── 2024-II/
│   └── [future courses...]
```

📦 **Lectures, Workshops, Quizzes, Announcements (per course)**
```
content/courses/2025-I/programming-iii/
├── index.md                  (course overview)
├── lectures/
│   ├── 01-intro.md          (date, slides, video, repo link)
│   └── 02-functions.md
├── workshops/
│   └── 01-design-patterns.md
├── quizzes/
│   └── 01-basics.md
└── announcements/
    └── 01-midterm.md        (date, important flag, content)
```

📦 **Projects**
```
content/projects/
├── basil-saas.md           (year, status, featured, categories, tech stack, links, role, description)
└── mercadoyepes.md
```

📦 **Publications**
```
content/publications/
├── purdue-dpcte.md         (year, venue, authors, type, links to paper/code/arxiv)
└── [future papers...]
```

📦 **Experience/Skills/Honors/Talks** (similar structure)

---

## 8. ARCHITECTURE RECOMMENDATION

### Option A: **Migrate to Astro** (Recommended)
- ✅ Static-first, built for content collections
- ✅ Native Markdown + frontmatter collections with Zod schema validation
- ✅ Component-based layout (reusable sections)
- ✅ No Node server required; GitHub Pages compatible
- ✅ Built-in image optimization, CSS scoping, minimal JS
- ❌ Requires build step + npm

### Option B: **Stay Static + Implement Lightweight Build** (Alternative)
- Use 11ty or Hugo to process Markdown → HTML
- Stays pure static; GitHub Pages works
- Less opinionated; shorter learning curve
- ❌ Manual routing; no built-in collection types

### Option C: **Keep Current HTML + Add Metadata**
- ❌ Doesn't solve hardcoded content problem
- ❌ No schema validation
- ❌ Content and presentation remain tightly coupled

**Recommendation: Option A (Astro)** — aligns with brief's content-driven philosophy, provides schema validation, keeps GitHub Pages deployment simple, and handles all cross-domain concerns (teaching + engineering + research in one site).

---

## 9. PHASE-BY-PHASE PLAN (If Astro Migration is Approved)

```
Phase 1:  Audit                           ← YOU ARE HERE
Phase 2:  Architecture proposal (waiting for approval)
Phase 3:  Astro scaffolding + npm setup
Phase 4:  Content schemas (Zod) + collection types
Phase 5:  Site config (branding, nav, social links)
Phase 6:  Design tokens + global styles
Phase 7:  Layout/nav/footer components
Phase 8:  Homepage (integrated, not two separate sections)
Phase 9:  Academic architecture (teaching, courses, semesters)
Phase 10: Course resources (lectures, workshops, quizzes, announcements)
Phase 11: Engineering/projects section
Phase 12: Research/publications section
Phase 13: About/experience/skills/honors sections
Phase 14: Responsive design pass
Phase 15: Accessibility pass
Phase 16: SEO/metadata pass
Phase 17: GitHub Pages build + deploy workflow (GitHub Actions)
Phase 18: Content templates + migration of existing content
Phase 19: README rewrite (setup, local dev, content authoring guide)
Phase 20: Final QA + cleanup
```

---

## 10. MISSING INFORMATION (You Must Provide)

Before proceeding, I need clarification on:

### Identity & Content
- [ ] Full current course list (all semesters) with course codes, titles, syllabi, active/featured flags
- [ ] All research publications (title, year, venue, authors, links to paper/arxiv/DOI/code/bibtex)
- [ ] List of all projects (with years, client/org names, tech stack, results, links)
- [ ] Experience history (employers, roles, dates, descriptions, locations)
- [ ] Education (degrees, institutions, years, focus areas)
- [ ] Skills (categories and technologies)
- [ ] Honors/awards/certifications (with years)
- [ ] Teaching plan structure (should it be generated from course metadata, or is Markmap the source of truth?)

### Branding & Navigation
- [ ] Preferred primary color (you provided `#42A5F6` in brief; confirm)
- [ ] Secondary colors if any
- [ ] Logo or wordmark for site header?
- [ ] Social/profile links (GitHub, LinkedIn, ORCID, Google Scholar, email, etc.) — with actual URLs
- [ ] Homepage sections (which should be featured? featured teaching? featured research? featured projects?)
- [ ] Navigation order/grouping (the brief suggests Academic / Engineering / Research / About; agree?)

### Courses & Teaching
- [ ] Repository links for each course (GitHub org? private/public?)
- [ ] Structure of course resources (lectures, workshops, quizzes, announcements — do all apply to all courses?)
- [ ] Are course pages meant to be browsable by outsiders, or students-only?

### Contact & Deployment
- [ ] Preferred contact method (email, contact form, or neither)?
- [ ] Custom domain or stay on `byepesg.github.io`?
- [ ] Any integrations needed (Disqus, analytics, CMS)?

---

## NEXT STEP

**Stop here.** I've completed Phase 1 (audit). Please review this report and:

1. **Confirm or adjust** the recommended architecture (Astro migration vs. alternatives)
2. **Provide the missing information** listed in Section 10
3. **State any hard constraints** (e.g., "must not use npm," "only pure HTML," etc.)

Once approved, I'll proceed to **Phase 2: Propose detailed architecture** (content schemas, site structure, component library), then await your second sign-off before writing code.

---

**Audit completed by:** AI Assistant | **Approved by:** [Awaiting your review]
