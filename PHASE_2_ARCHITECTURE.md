# PHASE 2: ARCHITECTURE PROPOSAL
**Date:** 2026-09-07 | **Status:** Complete — awaiting approval before Phase 3 implementation

---

## EXECUTIVE SUMMARY

This document proposes a content-driven static site architecture using **Astro** with structured content collections, schema validation, and automated GitHub Pages deployment. The site will serve as an integrated identity for teaching, research, engineering, and professional presence.

**Framework:** Astro (static-first, schema validation, minimal JS, GitHub Pages native)  
**Content:** Markdown + YAML frontmatter in `/src/content/` with Zod schema validation  
**Styling:** Tailwind CSS + CSS variables for design tokens  
**Deployment:** GitHub Actions workflow (auto-build on push to main)  
**Navigation:** `/` (Home) / `/academic` / `/engineering` / `/research` / `/about`

---

## 1. PROJECT STRUCTURE (TARGET)

```
src/
├── components/
│   ├── ui/                      # Reusable UI components
│   │   ├── Header.astro
│   │   ├── Footer.astro
│   │   ├── Nav.astro
│   │   ├── Card.astro
│   │   ├── Button.astro
│   │   ├── Badge.astro
│   │   └── Link.astro
│   ├── academic/                # Academic-specific components
│   │   ├── CourseCard.astro
│   │   ├── SemesterGroup.astro
│   │   ├── LectureList.astro
│   │   └── AnnouncementBanner.astro
│   ├── engineering/             # Engineering-specific components
│   │   ├── ProjectCard.astro
│   │   └── ProjectGrid.astro
│   ├── research/                # Research-specific components
│   │   ├── PublicationCard.astro
│   │   └── PublicationList.astro
│   └── home/                    # Homepage components
│       ├── Hero.astro
│       ├── Featured.astro
│       └── PathwaySection.astro
│
├── content/
│   ├── config.ts                # Schema definitions (Zod)
│   ├── site.yml                 # Global site config
│   ├── courses/                 # Teaching content
│   │   ├── 2025-I/
│   │   │   ├── programming-iii.md
│   │   │   ├── databases-iii.md
│   │   │   └── data-structures.md
│   │   └── 2024-II/
│   │       └── [future courses]
│   ├── lectures/                # Lectures (auto-linked to courses)
│   │   ├── 2025-I-programming-iii-01.md
│   │   ├── 2025-I-programming-iii-02.md
│   │   └── [more lectures]
│   ├── projects/
│   │   ├── basil-saas.md
│   │   └── mercadoyepes.md
│   ├── publications/
│   │   ├── purdue-dpcte.md
│   │   └── [future papers]
│   ├── experience/
│   │   ├── professor-at-unal.md
│   │   ├── researcher-at-purdue.md
│   │   └── [past roles]
│   ├── skills/
│   │   ├── programming-languages.md
│   │   ├── frameworks.md
│   │   └── research-methods.md
│   └── honors/
│       ├── award-2024.md
│       └── [recognition items]
│
├── layouts/
│   ├── Base.astro               # Main layout wrapper
│   ├── PageLayout.astro         # Standard page (with header/footer)
│   ├── CourseLayout.astro       # Course detail page
│   ├── ProjectLayout.astro      # Project case study
│   └── PublicationLayout.astro  # Publication detail
│
├── pages/
│   ├── index.astro              # Homepage
│   ├── academic/
│   │   ├── index.astro          # Academic hub
│   │   ├── teaching/
│   │   │   ├── index.astro      # All courses (grouped by semester)
│   │   │   └── [slug].astro     # Individual course
│   │   └── [other academic routes]
│   ├── engineering/
│   │   ├── index.astro          # Engineering hub
│   │   ├── projects/
│   │   │   ├── index.astro      # All projects
│   │   │   └── [slug].astro     # Project detail
│   │   └── [other engineering routes]
│   ├── research/
│   │   ├── index.astro          # Research hub
│   │   ├── publications/
│   │   │   ├── index.astro      # Publications (sorted newest-first, grouped by year)
│   │   │   └── [slug].astro     # Publication detail
│   │   └── [other research routes]
│   ├── about/
│   │   ├── index.astro          # About + bio + profile links
│   │   ├── experience.astro     # Career history
│   │   ├── skills.astro         # Technical skills
│   │   └── honors.astro         # Awards + recognition
│   ├── 404.astro                # Not found page
│   └── robots.txt.ts            # SEO
│
├── config/
│   ├── site.ts                  # Site metadata (name, title, social links, nav)
│   ├── colors.ts                # Design tokens (colors, spacing, typography)
│   └── constants.ts             # Constants (semester codes, etc.)
│
├── styles/
│   ├── global.css               # Global resets, vars, typography
│   ├── tailwind.config.js       # Tailwind config (theme overrides)
│   └── [component-scoped CSS as needed]
│
├── utils/
│   ├── date.ts                  # Date formatting helpers
│   ├── content.ts               # Content filtering/sorting helpers
│   └── seo.ts                   # SEO metadata generation
│
└── env.d.ts                     # TypeScript definitions
│
public/
├── images/
│   ├── projects/                # Project images/GIFs
│   │   ├── basil-saas-1.gif
│   │   ├── basil-saas-2.gif
│   │   ├── mercadoyepes-1.gif
│   │   └── mercadoyepes-2.gif
│   ├── icons/                   # Social icons, badges
│   └── og/                      # OG image templates
│
root/
├── astro.config.mjs             # Astro config
├── tsconfig.json                # TypeScript config
├── tailwind.config.cjs          # Tailwind config
├── .env.example                 # Environment variables template
├── package.json                 # Dependencies + scripts
├── PROGRESS.md                  # (existing) Session tracking
├── PHASE_1_AUDIT.md             # (existing) Audit report
├── PHASE_2_ARCHITECTURE.md      # This file
├── README.md                    # (to be rewritten) Setup + maintenance guide
└── .github/
    └── workflows/
        └── deploy.yml           # GitHub Actions: validate build + deploy
```

---

## 2. CONTENT SCHEMAS (Zod Validation)

### 2.1 Course Schema

```typescript
// src/content/config.ts
import { defineCollection, z } from 'astro:content';

const courses = defineCollection({
  schema: z.object({
    code: z.string(),                    // e.g., "PROG-III"
    title: z.string(),                   // e.g., "Programming III"
    semester: z.string(),                // e.g., "2025-I"
    description: z.string(),
    syllabus_url: z.string().url().optional(),
    repository_url: z.string().url().optional(),
    credits: z.number().optional(),
    prerequisites: z.array(z.string()).optional(),
    active: z.boolean().default(false),  // "Current Teaching"
    featured: z.boolean().default(false),// Homepage feature
    published: z.boolean().default(true),
    instructors: z.array(z.string()).optional(),
    office_hours: z.string().optional(),
  }),
});

const lectures = defineCollection({
  schema: z.object({
    course_slug: z.string(),              // e.g., "programming-iii"
    semester: z.string(),                 // e.g., "2025-I"
    number: z.number(),                   // Lecture #1, #2, etc.
    title: z.string(),
    date: z.date().optional(),
    topics: z.array(z.string()).optional(),
    slides_url: z.string().url().optional(),
    video_url: z.string().url().optional(),
    code_repo_url: z.string().url().optional(),
    reading_urls: z.array(z.object({
      title: z.string(),
      url: z.string().url(),
    })).optional(),
    published: z.boolean().default(true),
  }),
});

const projects = defineCollection({
  schema: z.object({
    title: z.string(),
    slug: z.string(),
    year: z.number(),
    status: z.enum(['completed', 'active', 'archived']),
    featured: z.boolean().default(false),
    published: z.boolean().default(true),
    categories: z.array(z.enum([
      'AI/ML',
      'Software Engineering',
      'Research Software',
      'Backend',
      'Frontend',
      'Algorithms',
      'Open Source',
      'Web App',
    ])),
    technologies: z.array(z.string()),
    client_org: z.string().optional(),
    description: z.string(),
    role: z.string().optional(),
    outcomes: z.array(z.string()).optional(),
    links: z.object({
      github: z.string().url().optional(),
      demo: z.string().url().optional(),
      paper: z.string().url().optional(),
      docs: z.string().url().optional(),
    }).optional(),
    image_url: z.string().optional(),
    image_alt: z.string().optional(),
  }),
});

const publications = defineCollection({
  schema: z.object({
    title: z.string(),
    slug: z.string(),
    year: z.number(),
    type: z.enum(['journal', 'conference', 'preprint', 'workshop', 'book']),
    venue: z.string(),                    // Journal name or conference name
    featured: z.boolean().default(false),
    published: z.boolean().default(true),
    authors: z.array(z.string()),         // "Jane Doe, John Smith"
    description: z.string().optional(),
    abstract: z.string().optional(),
    links: z.object({
      paper: z.string().url().optional(),
      arxiv: z.string().url().optional(),
      doi: z.string().optional(),
      code: z.string().url().optional(),
      bibtex: z.string().optional(),
    }).optional(),
  }),
});

// Export collections for Astro
export const collections = { courses, lectures, projects, publications };
```

### 2.2 Site Config (YAML)

```yaml
# src/content/site.yml
site:
  title: "Sebastián Yepes"
  tagline: "Professor · Researcher · Software Engineer"
  description: "Academic and engineering portfolio of Sebastián Yepes, professor at UNAL."
  
  author:
    name: "Sebastián Yepes García"
    bio: "Professor of computer science at UNAL, specializing in algorithms, data structures, and software engineering. Researcher in privacy-preserving data compression."
    email: "sebastian.yepes@unal.edu.co"
    avatar_url: "/images/avatars/sebastian.jpg"  # TODO: Add actual photo
    
  social:
    github: "https://github.com/byepesg"
    linkedin: "https://linkedin.com/in/byepesg"
    twitter: "https://twitter.com/byepesg"  # Optional
    orcid: "https://orcid.org/0000-0000-0000-0000"  # TODO: Add ORCID
    scholar: "https://scholar.google.com/citations?user=..."  # TODO: Add Scholar ID
    email: "sebastian.yepes@unal.edu.co"

  nav:
    - label: "Home"
      href: "/"
    - label: "Academic"
      href: "/academic"
      submenu:
        - label: "Teaching"
          href: "/academic/teaching"
        - label: "Research"
          href: "/research"
    - label: "Engineering"
      href: "/engineering"
      submenu:
        - label: "Projects"
          href: "/engineering/projects"
    - label: "About"
      href: "/about"

  footer:
    copyright: "© 2024 Sebastián Yepes. All rights reserved."
    links:
      - label: "GitHub"
        href: "https://github.com/byepesg"
      - label: "Email"
        href: "mailto:sebastian.yepes@unal.edu.co"
```

---

## 3. EXAMPLE CONTENT FILES (Mock Data)

### 3.1 Course Example

```markdown
// src/content/courses/2025-I/programming-iii.md
---
code: "PROG-III"
title: "Programming III"
semester: "2025-I"
description: "Advanced topics in object-oriented programming, design patterns, and clean code principles."
syllabus_url: "https://github.com/byepesg/programming-iii-2025-I/blob/main/README.md"
repository_url: "https://github.com/byepesg/programming-iii-2025-I"
credits: 4
prerequisites: ["Programming II", "Data Structures"]
active: true
featured: true
instructors: ["Sebastián Yepes García"]
office_hours: "Tuesdays 2-4 PM, Building C Room 302"
---

## Overview
This course explores advanced programming concepts including design patterns, refactoring, testing, and software architecture. Students will implement real-world projects using best practices.

## Learning Outcomes
- Understand and apply common design patterns
- Write maintainable, testable code
- Refactor legacy code
- Work effectively in teams

## Course Materials
- Textbook: "Clean Code" by Robert C. Martin
- Repository: All code examples and assignments in [GitHub](https://github.com/byepesg/programming-iii-2025-I)

## Grading
- Assignments: 40%
- Midterm Project: 30%
- Final Project: 30%
```

### 3.2 Lecture Example

```markdown
// src/content/lectures/2025-I-programming-iii-01.md
---
course_slug: "programming-iii"
semester: "2025-I"
number: 1
title: "Introduction to Design Patterns"
date: 2025-01-22
topics: ["Design Patterns", "OOP Principles", "Creational Patterns"]
slides_url: "https://github.com/byepesg/programming-iii-2025-I/tree/main/lectures/01-slides.pdf"
video_url: "https://youtube.com/watch?v=..."  # TODO: Add after recording
code_repo_url: "https://github.com/byepesg/programming-iii-2025-I/tree/main/lectures/01-code"
reading_urls:
  - title: "Design Patterns: Elements of Reusable Object-Oriented Software"
    url: "https://en.wikipedia.org/wiki/Design_Patterns"
  - title: "Creational Patterns (Wikipedia)"
    url: "https://en.wikipedia.org/wiki/Creational_pattern"
---

## Lecture 1: Introduction to Design Patterns

### What are Design Patterns?
Design patterns are proven solutions to common programming problems...

### Creational Patterns Overview
We'll cover Singleton, Factory, Abstract Factory, Builder...

### Code Examples
See the repository for all code examples used in this lecture.
```

### 3.3 Project Example

```markdown
// src/content/projects/basil-saas.md
---
title: "Basil SaaS Platform"
slug: "basil-saas"
year: 2023
status: "completed"
featured: true
categories: ["Software Engineering", "Web App", "Frontend"]
technologies: ["Vue.js", "Laravel", "PostgreSQL", "AWS"]
client_org: "Basil S.A.S."
description: "Enterprise SaaS platform for agricultural data management and analytics"
role: "Lead Software Engineer"
outcomes:
  - "Delivered production system serving 500+ users"
  - "Reduced data processing time by 60% through optimization"
  - "Mentored 3 junior developers"
links:
  demo: "https://basil-saas.example.com"
  github: "https://github.com/basil-org/platform-private"  # (if public)
image_url: "/images/projects/basil-saas-1.gif"
image_alt: "Basil dashboard showing crop analytics"
---

## Project Overview
Basil is a SaaS platform designed to help agricultural companies manage crop data, optimize resources, and make data-driven decisions.

## Challenge
The client needed a system to:
- Ingest data from IoT sensors in the field
- Process and visualize agronomic metrics
- Generate actionable insights for farm managers

## Solution
Built a Vue.js frontend with real-time dashboards, a Laravel backend API, and PostgreSQL data warehouse. Deployed on AWS with auto-scaling.

## Technical Architecture
- **Frontend:** Vue.js + Vuex, responsive design
- **Backend:** Laravel REST API, caching layer with Redis
- **Database:** PostgreSQL with optimized queries
- **Infrastructure:** AWS EC2, S3, CloudFront

## Results
- ✅ Launched in 6 months
- ✅ 500+ active users within first year
- ✅ 99.9% uptime SLA
- ✅ 40% reduction in client operational costs

## Role & Impact
Led a team of 3 engineers, designed system architecture, mentored junior developers in best practices, and managed client communication.
```

### 3.4 Publication Example

```markdown
// src/content/publications/purdue-dpcte.md
---
title: "Differential Privacy in Compress-Then-Encrypt Schemes"
slug: "purdue-dpcte"
year: 2024
type: "preprint"
venue: "Under Review for STOC"
featured: true
authors:
  - "Sebastián Yepes García"
  - "Jeremiah M. Blocki"
  - "Seunghoon Lee"
abstract: "We design and analyze DPCTE schemes that combine data compression and encryption while maintaining differential privacy guarantees. Our construction achieves O(log n) overhead over standard compression, with applications to privacy-preserving data analytics."
links:
  arxiv: "https://arxiv.org/abs/2024.xxxxx"
  paper: "https://example.com/papers/dpcte-2024.pdf"
  code: "https://github.com/byepesg/dpcte-schemes"
---

## Summary

This work addresses the problem of combining compression and encryption while maintaining differential privacy. Traditional approaches suffer from high overhead; we propose a novel scheme achieving near-optimal bounds.

## Key Contributions

1. **Theoretical Analysis:** Proof of lower bounds for DP compression-encryption trade-offs
2. **Algorithm:** Construction achieving O(log n) overhead
3. **Implementation:** Practical evaluation on real datasets

## Motivation

Privacy-preserving compression is critical for cloud storage and analytics applications where data must be both compact and protected.

## Related Work

We build on prior work in differential privacy and information-theoretic compression...

## References

See the paper for full references.
```

---

## 4. SITE ROUTING & NAVIGATION

| Route | Component | Data Source | Purpose |
|-------|-----------|-------------|---------|
| `/` | `pages/index.astro` | All collections | Homepage — integrated landing |
| `/academic` | `pages/academic/index.astro` | Courses, research intro | Academic hub |
| `/academic/teaching` | `pages/academic/teaching/index.astro` | Courses (sorted by semester) | All courses grouped by semester |
| `/academic/teaching/[slug]` | `pages/academic/teaching/[slug].astro` | Single course + related lectures | Course detail page |
| `/research` | `pages/research/index.astro` | Publications intro | Research hub |
| `/research/publications` | `pages/research/publications/index.astro` | Publications (sorted newest-first, grouped by year) | Publication list |
| `/research/publications/[slug]` | `pages/research/publications/[slug].astro` | Single publication | Publication detail |
| `/engineering` | `pages/engineering/index.astro` | Projects intro | Engineering hub |
| `/engineering/projects` | `pages/engineering/projects/index.astro` | Projects (sorted featured-first, then newest) | Project list |
| `/engineering/projects/[slug]` | `pages/engineering/projects/[slug].astro` | Single project | Project case study |
| `/about` | `pages/about/index.astro` | Site config + author bio | About + profile links |
| `/about/experience` | `pages/about/experience.astro` | Experience collection | Career history |
| `/about/skills` | `pages/about/skills.astro` | Skills collection | Technical skills |
| `/about/honors` | `pages/about/honors.astro` | Honors collection | Awards + recognition |
| `/sitemap.xml` | `pages/sitemap.xml.ts` | All collections | SEO sitemap |
| `/robots.txt` | `pages/robots.txt.ts` | Static | SEO robots file |

---

## 5. DESIGN SYSTEM & TOKENS

### 5.1 Color Palette

```typescript
// src/config/colors.ts
export const colors = {
  // Brand
  primary: '#42A5F6',           // Blue (links, CTAs, badges)
  
  // Backgrounds
  bg_white: '#FFFFFF',
  bg_light: '#F8FAFC',
  bg_lighter: '#F1F5F9',
  
  // Borders & Dividers
  border_light: '#E2E8F0',
  border_medium: '#CBD5E1',
  border_dark: '#94A3B8',
  
  // Text
  text_primary: '#0F172A',      // Almost black
  text_secondary: '#475569',    // Dark gray
  text_tertiary: '#64748B',     // Medium gray
  text_muted: '#94A3B8',        // Light gray
  text_inverse: '#FFFFFF',      // White (on dark)
  
  // Semantic
  success: '#10B981',
  warning: '#F59E0B',
  danger: '#EF4444',
  info: '#3B82F6',
};
```

### 5.2 Typography

```typescript
// src/config/typography.ts
export const typography = {
  fonts: {
    sans: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    mono: '"Fira Code", "Source Code Pro", monospace',
  },
  
  sizes: {
    xs: '0.75rem',    // 12px
    sm: '0.875rem',   // 14px
    base: '1rem',     // 16px
    lg: '1.125rem',   // 18px
    xl: '1.25rem',    // 20px
    '2xl': '1.5rem',  // 24px
    '3xl': '1.875rem', // 30px
    '4xl': '2.25rem', // 36px
  },
  
  weights: {
    normal: 400,
    semibold: 600,
    bold: 700,
  },
};
```

### 5.3 Spacing Scale

```css
/* src/styles/global.css */
:root {
  --space-0: 0;
  --space-1: 0.25rem;  /* 4px */
  --space-2: 0.5rem;   /* 8px */
  --space-3: 0.75rem;  /* 12px */
  --space-4: 1rem;     /* 16px */
  --space-6: 1.5rem;   /* 24px */
  --space-8: 2rem;     /* 32px */
  --space-12: 3rem;    /* 48px */
  --space-16: 4rem;    /* 64px */
  --space-24: 6rem;    /* 96px */
}
```

---

## 6. HOMEPAGE DESIGN

### 6.1 Structure (Wireframe)

```
┌─────────────────────────────────────────┐
│ Header / Nav                            │
├─────────────────────────────────────────┤
│ Hero Section                            │
│ ┌─────────────────────────────────────┐ │
│ │ Sebastián Yepes                     │ │
│ │ Professor · Researcher · Engineer   │ │
│ │ [Research highlights / current focus] │
│ │ [Social links: GitHub, Scholar, ...]│
│ └─────────────────────────────────────┘ │
├─────────────────────────────────────────┤
│ Pathways (Two Columns)                  │
│ ┌──────────────┐  ┌──────────────┐      │
│ │ Academic     │  │ Engineering  │      │
│ │ Teaching     │  │ Projects     │      │
│ │ Research     │  │ [CTA buttons]│      │
│ └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────┤
│ Featured Courses (if active=true)       │
│ ┌────────────┐ ┌────────────┐           │
│ │ PROG-III   │ │ DATA-STR   │           │
│ │ (2025-I)   │ │ (2025-I)   │           │
│ └────────────┘ └────────────┘           │
├─────────────────────────────────────────┤
│ Featured Projects                       │
│ ┌────────────────────────────────┐      │
│ │ Basil SaaS Platform            │      │
│ │ [Description + link]            │      │
│ └────────────────────────────────┘      │
├─────────────────────────────────────────┤
│ Latest Publications                     │
│ [List of 3 most recent papers]         │
├─────────────────────────────────────────┤
│ Footer                                  │
└─────────────────────────────────────────┘
```

### 6.2 Content Rules

- **Hero:** Fixed name, title, bio (from site config)
- **Courses:** Pull all where `active: true`, limit to 3 most recent semesters
- **Projects:** Pull all where `featured: true`, sorted newest-first; fallback to first 3 if none featured
- **Publications:** Pull newest 3 where `featured: true` or all newest 3 if none featured
- **Sections:** If no active courses → hide course section; if no featured projects → hide projects section (never render empty slots)

---

## 7. COMPONENT EXAMPLES

### 7.1 CourseCard Component

```astro
// src/components/academic/CourseCard.astro
---
interface Props {
  code: string;
  title: string;
  semester: string;
  active: boolean;
  featured: boolean;
  description: string;
  slug: string;
}

const { code, title, semester, active, featured, description, slug } = Astro.props;
---

<a href={`/academic/teaching/${slug}`} class="course-card">
  <div class="course-header">
    <h3 class="course-title">{title}</h3>
    <div class="course-meta">
      <span class="course-code">{code}</span>
      <span class="semester">{semester}</span>
      {active && <span class="badge badge-active">Current</span>}
      {featured && <span class="badge badge-featured">Featured</span>}
    </div>
  </div>
  <p class="course-description">{description}</p>
  <span class="link-cta">View Course →</span>
</a>

<style>
  .course-card {
    display: block;
    padding: var(--space-6);
    border: 1px solid var(--color-border-light);
    border-radius: 8px;
    text-decoration: none;
    transition: all 0.2s ease;
  }

  .course-card:hover {
    border-color: var(--color-primary);
    box-shadow: 0 2px 8px rgba(66, 165, 246, 0.1);
  }

  .course-title {
    font-size: var(--size-lg);
    font-weight: 600;
    margin: 0 0 var(--space-2) 0;
    color: var(--color-text-primary);
  }

  .course-meta {
    display: flex;
    gap: var(--space-2);
    flex-wrap: wrap;
    margin-bottom: var(--space-3);
  }

  .course-code {
    font-size: var(--size-sm);
    font-weight: 600;
    color: var(--color-primary);
    font-family: var(--font-mono);
  }

  .semester {
    font-size: var(--size-sm);
    color: var(--color-text-tertiary);
  }

  .badge {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
  }

  .badge-active {
    background-color: rgba(16, 185, 129, 0.1);
    color: #059669;
  }

  .badge-featured {
    background-color: rgba(66, 165, 246, 0.1);
    color: var(--color-primary);
  }

  .course-description {
    margin: 0 0 var(--space-4) 0;
    font-size: var(--size-sm);
    color: var(--color-text-secondary);
    line-height: 1.6;
  }

  .link-cta {
    display: inline-block;
    font-size: var(--size-sm);
    font-weight: 600;
    color: var(--color-primary);
    text-decoration: none;
    transition: transform 0.2s ease;
  }

  .course-card:hover .link-cta {
    transform: translateX(2px);
  }
</style>
```

### 7.2 Header Component

```astro
// src/components/ui/Header.astro
---
import { site } from "@/config/site";

interface Props {
  currentPath?: string;
}

const { currentPath = "/" } = Astro.props;
---

<header class="site-header">
  <div class="header-container">
    <a href="/" class="logo">
      <span class="logo-text">{site.author.name}</span>
    </a>
    <nav class="main-nav">
      {site.nav.map((item) => (
        <a
          href={item.href}
          class={`nav-link ${currentPath === item.href ? "active" : ""}`}
        >
          {item.label}
        </a>
      ))}
    </nav>
    <div class="social-links">
      <a href={site.social.github} title="GitHub" aria-label="GitHub" target="_blank">
        <svg class="icon"><!-- GitHub icon --></svg>
      </a>
      {site.social.linkedin && (
        <a href={site.social.linkedin} title="LinkedIn" aria-label="LinkedIn" target="_blank">
          <svg class="icon"><!-- LinkedIn icon --></svg>
        </a>
      )}
    </div>
  </div>
</header>

<style>
  .site-header {
    background-color: var(--color-bg-white);
    border-bottom: 1px solid var(--color-border-light);
    position: sticky;
    top: 0;
    z-index: 100;
  }

  .header-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: var(--space-4);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .logo {
    font-size: var(--size-lg);
    font-weight: 700;
    color: var(--color-text-primary);
    text-decoration: none;
  }

  .main-nav {
    display: flex;
    gap: var(--space-8);
  }

  .nav-link {
    font-size: var(--size-base);
    font-weight: 500;
    color: var(--color-text-secondary);
    text-decoration: none;
    transition: color 0.2s ease;
  }

  .nav-link:hover,
  .nav-link.active {
    color: var(--color-primary);
  }

  .social-links {
    display: flex;
    gap: var(--space-4);
  }

  .icon {
    width: 20px;
    height: 20px;
    color: var(--color-text-secondary);
    transition: color 0.2s ease;
  }

  .social-links a:hover .icon {
    color: var(--color-primary);
  }

  @media (max-width: 768px) {
    .header-container {
      flex-wrap: wrap;
    }

    .main-nav {
      gap: var(--space-4);
      order: 3;
      width: 100%;
      margin-top: var(--space-4);
    }
  }
</style>
```

---

## 8. BUILD & DEPLOYMENT WORKFLOW

### 8.1 Local Development

```bash
# Install dependencies
npm install

# Start dev server (http://localhost:3000)
npm run dev

# Build for production
npm run build

# Preview production build locally
npm run preview

# Lint + type-check
npm run lint
npm run typecheck
```

### 8.2 GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Build and Deploy

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: "18"
          cache: "npm"

      - name: Install dependencies
        run: npm ci

      - name: Lint
        run: npm run lint

      - name: Type check
        run: npm run typecheck

      - name: Validate content (frontmatter + schemas)
        run: npm run validate:content

      - name: Build
        run: npm run build

      - name: Check for broken links
        run: npm run check:links || true  # Optional, don't fail on link checks

      - name: Deploy to GitHub Pages
        if: github.ref == 'refs/heads/main' && github.event_name == 'push'
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./dist
```

### 8.3 package.json Scripts

```json
{
  "scripts": {
    "dev": "astro dev",
    "build": "astro build",
    "preview": "astro preview",
    "lint": "eslint src --ext .ts,.astro && prettier --check src",
    "lint:fix": "eslint src --ext .ts,.astro --fix && prettier --write src",
    "typecheck": "astro check",
    "validate:content": "node scripts/validate-content.js",
    "check:links": "astro build && linkinator dist"
  },
  "dependencies": {
    "astro": "^4.0.0",
    "@astrojs/react": "^3.0.0",
    "@astrojs/tailwind": "^0.0.0",
    "tailwindcss": "^3.0.0",
    "zod": "^3.0.0"
  },
  "devDependencies": {
    "@astrojs/check": "^0.0.0",
    "astro-icon": "^0.0.0",
    "eslint": "^8.0.0",
    "prettier": "^3.0.0",
    "typescript": "^5.0.0"
  }
}
```

---

## 9. CONTENT MIGRATION CHECKLIST (Phase 18)

### From Current Hardcoded HTML → Structured Markdown

- [ ] **Courses:** Extract 3 courses from teaching.html → individual .md files
  - [ ] Add syllabus URLs (if available)
  - [ ] Add repository URLs
  - [ ] Set `active` flag for current semester
  - [ ] Set `featured` flag for homepage display

- [ ] **Lectures:** Create lecture entries for each course
  - [ ] Link to GitHub repos (slides, code)
  - [ ] Add video links (if available)
  - [ ] Organize by course + semester

- [ ] **Projects:** Extract 2 projects from projects.html → .md files
  - [ ] Preserve images (move to `/public/images/projects/`)
  - [ ] Add actual GitHub links / demo links
  - [ ] Add year, status, categories, tech stack
  - [ ] Set `featured: true` for homepage

- [ ] **Publications:** Create publication entry for Purdue research
  - [ ] Extract authors, venue, year
  - [ ] Add arXiv/paper links
  - [ ] Add GitHub repo if code is public
  - [ ] Set `featured: true`

- [ ] **Experience:** Create entries for past roles
  - [ ] UNAL professor position
  - [ ] Purdue internship
  - [ ] Any other roles

- [ ] **Skills:** Organize by category
  - [ ] Programming languages
  - [ ] Frameworks (Vue, Laravel, React, etc.)
  - [ ] Tools/platforms (AWS, Git, Docker, etc.)
  - [ ] Research methods

- [ ] **About:** Write professional bio + narrative

---

## 10. MISSING INFORMATION (You Must Provide Later)

| Item | Purpose | Example |
|------|---------|---------|
| Courses (all semesters) | Teaching hub | PROG-III, DATABASES-III, DATA-STR, etc. |
| Lectures (full list) | Course detail pages | With dates, links, videos |
| All projects (actual list) | Engineering portfolio | Years, links, descriptions |
| All publications | Research section | Titles, venues, DOIs, arXiv IDs |
| Experience history | About/career page | Employers, dates, roles, descriptions |
| Skills taxonomy | Skills page | Categories + technologies |
| Honors/awards | Recognition section | Dates, issuing org, descriptions |
| Photo / avatar | Header + about | 200x200px JPG or PNG |
| Social links | Header + footer | GitHub, LinkedIn, ORCID, Scholar profiles |
| Bio/tagline | Homepage | 1-2 sentence professional statement |
| Teaching philosophy | Academic page | Optional narrative |
| Research statement | Research page | Optional narrative |

---

## 11. NEXT STEPS (If Approved)

1. **Phase 3:** Astro scaffolding + npm setup
2. **Phase 4:** Content schemas (Zod) + collection types
3. **Phase 5:** Site config (branding, nav, social links)
4. **Phase 6:** Design tokens + global styles
5. **Phase 7–17:** Component build + page implementation
6. **Phase 18:** Migrate existing content into structured format
7. **Phase 19:** Rewrite README with setup/maintenance/authoring guide
8. **Phase 20:** Final QA + cleanup

---

## ARCHITECTURE SUMMARY

| Aspect | Solution |
|--------|----------|
| **Framework** | Astro (static-first, schema validation, GitHub Pages compatible) |
| **Content Model** | Markdown + YAML frontmatter + Zod validation |
| **Styling** | Tailwind CSS + CSS variables (design tokens) |
| **State Management** | None needed (static generation) |
| **SEO** | Automatic (Astro meta + structured data) |
| **Build** | npm run build → /dist (GitHub Pages) |
| **Deployment** | GitHub Actions (validate + build + deploy on main push) |
| **Maintenance** | Markdown editing + rebuild (no database, no server) |

---

**This document is a comprehensive blueprint. Review it and confirm:**

1. ✅ **Architecture approved?** (Astro, content structure, routes, design system)
2. ✅ **Ready to proceed to Phase 3?** (Scaffolding + setup)
3. ❓ **Any changes or clarifications needed?**

Once approved, I'll begin Phase 3 implementation.
