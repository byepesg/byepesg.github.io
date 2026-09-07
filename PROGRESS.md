# PROJECT PROGRESS TRACKER

**Project:** Rebuild of Sebastián Yepes' Personal Site (Academic + Engineering)  
**Repository:** `byepesg.github.io` (GitHub Pages)  
**Started:** 2026-09-07

---

## Current Status: PHASE 1 COMPLETE — AWAITING APPROVAL

### Phase 1: Audit ✅
**Completed:** 2026-09-07  
**Deliverable:** `PHASE_1_AUDIT.md`

**Key Findings:**
- Current tech: Pure static HTML + Tailwind CDN (no build tool)
- Content: Hardcoded courses (3), projects (2), publications (1 real + 2 templates)
- Structure: 13 section fragments + 3 index variants (2 unused)
- GitHub Pages: No CI/CD, no validation, manual push deployment
- Dead ends: index2.html, index3.html; broken image paths; external CDN deps

**Recommended Architecture:** Astro (static-first, content-driven, validates schemas, GitHub Pages compatible)

**Open Questions:**
1. Approve Astro migration, or use alternative?
2. Provide missing content info (see audit Section 10)
3. Any hard constraints (npm restrictions, etc.)?

---

### Phase 2: Architecture Proposal ⏳
**Status:** Not started (awaiting Phase 1 approval)  
**Will include:**
- Confirmed framework choice
- Content collection schemas (courses, projects, publications, etc.)
- Site structure (routes, layouts, components)
- Design system (colors, spacing, typography)
- Build + deployment workflow

---

### Phase 3–20: Implementation
**Status:** Not started (depends on Phase 2 approval)

---

## Checkpoint Decisions & Rationale

| Decision | Reasoning | Status |
|----------|-----------|--------|
| Astro (proposed) | Content-driven, schema validation, GitHub Pages compatible, minimal JS, native Markdown support | Pending approval |
| Preserve existing projects & images | Real client work; valuable context for portfolio | Approved (assumed) |
| Refactor courses → structured metadata | Enable dynamic course pages, semesters, active/featured flags | Pending Phase 2 |
| Remove index2/3.html | Dead code; confusing; no active use detected | Pending Phase 2 |
| Centralize site config | Avoid duplicate profile info, social links, nav structure | Pending Phase 2 |

---

## Files Created/Modified

| File | Phase | Status | Notes |
|------|-------|--------|-------|
| `PHASE_1_AUDIT.md` | 1 | ✅ Created | Comprehensive audit report |
| `PROGRESS.md` | 1 | ✅ Created | This file; session tracking |
| `README.md` | 1 | ⚠️ Need rewrite | Currently only test comment; will document setup, dev, deployment, content authoring |
| `.github/workflows/` | 17 | 📋 Pending | GitHub Actions for build + deploy |

---

## Next Session Checklist

Before proceeding:
- [ ] Sebastián reviews PHASE_1_AUDIT.md
- [ ] Sebastián confirms architecture choice (Astro vs. alternative)
- [ ] Sebastián provides missing content (Section 10 of audit)
- [ ] Sebastián clarifies navigation structure & branding preferences
- [ ] AI Assistant starts Phase 2 (architecture proposal)

---

## Build & Deploy Status

| Aspect | Current | Target |
|--------|---------|--------|
| **Local build** | None (static HTML only) | `npm run build` (Astro) |
| **Dev server** | None | `npm run dev` (Astro) |
| **Production build** | None | GitHub Actions workflow |
| **Deployment** | Manual `git push` | Automated (GH Actions on push to main) |
| **Validation** | None | Pre-build schema validation + link checking |

---

## Known Blockers

1. **No content metadata:** Courses, projects, publications are hardcoded HTML; can't filter/sort/feature dynamically
2. **No build automation:** All changes require manual HTML editing and push; high error risk
3. **No SEO infrastructure:** No sitemaps, structured data, or metadata optimization
4. **Image paths fragile:** `/../Projects/Images/` relative paths may break

---

## Outstanding Questions for Sebastián

1. Framework choice: Astro, Hugo, 11ty, or stay static + lightweight build?
2. Domain: Keep `byepesg.github.io` or use custom domain (and which one)?
3. Teaching: Should course pages be public, or gated for students?
4. Semesters: List all semesters you teach (to organize courses)
5. Publications: Are there more papers/preprints beyond the Purdue research?
6. Featured content: Which projects, papers, courses should be highlighted on homepage?
7. Navigation: Do you want "Academic" and "Engineering" as separate top-level sections, or integrated?
8. Social links: GitHub, LinkedIn, ORCID, Google Scholar profiles — what should be linked?

---

## Session Log

### Session 1 (2026-09-07)
- ✅ Phase 1 audit completed
- ✅ PHASE_1_AUDIT.md and PROGRESS.md created
- ⏳ Awaiting Sebastián's feedback on architecture + missing content
- 🔗 Test push successful (via HTTPS + new SSH key setup)

### Session 2 [Pending]
- Awaiting review of audit report
- Will start Phase 2 once architecture is approved

---

## Maintenance Notes

- **Git branch strategy:** Once approved, work on feature branch (e.g., `redesign/content-architecture`), not main, until Phase 17 passes.
- **Build artifacts:** Will use `/dist/` for Astro output (if Astro approved); add to .gitignore.
- **Content updates:** All content will be in Markdown + frontmatter in `content/` directory; update PROGRESS.md + commit after each phase.
- **Deploy workflow:** Will add GitHub Actions workflow in Phase 17 to automate build + deploy on `main` push.

---

**Last Updated:** 2026-09-07 by AI Assistant  
**Next Update:** After Phase 1 review and approval
