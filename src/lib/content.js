import fs from 'node:fs';
import path from 'node:path';
import { validateProject } from './schemas/projectSchema.js';
import { validateCourse } from './schemas/courseSchema.js';

const projectsDir = path.resolve('./src/data/projects');
const projectContentDir = path.resolve('./src/content/projects');
const coursesDir = path.resolve('./src/content/courses');
const publicationsDir = path.resolve('./src/content/publications');
const researchDir = path.resolve('./src/content/research');
const academicDir = path.resolve('./src/content/academic');
const innovatrixDir = path.resolve('./src/content/innovatrix');

function readJsonFiles(dir) {
  try {
    if (!fs.existsSync(dir)) return [];
    return fs
      .readdirSync(dir)
      .filter(file => file.endsWith('.json'))
      .map(file => {
        const fullPath = path.join(dir, file);
        return {
          ...JSON.parse(fs.readFileSync(fullPath, 'utf-8')),
          _source: fullPath,
          _fileName: file
        };
      });
  } catch (error) {
    return [];
  }
}

function ensureSlug(entry) {
  if (!entry || typeof entry !== 'object') return entry;
  const fileName = entry._fileName || '';
  const fallback = fileName.replace(/\.[^.]+$/, '').replace(/[_\s]+/g, '-');
  return {
    ...entry,
    slug: entry.slug || fallback
  };
}

export async function getAllProjects() {
  const data = [...readJsonFiles(projectsDir), ...readJsonFiles(projectContentDir)];
  return data.map(item => validateProject(ensureSlug(item)));
}

export async function getAllCourses() {
  const data = readJsonFiles(coursesDir);
  return data.map(item => validateCourse(ensureSlug(item)));
}

export async function getAcademicProfile() {
  const data = readJsonFiles(academicDir);
  const profile = data[0] || {
    title: 'Academic profile',
    summary: 'Academic background in engineering, physics, and applied AI research.',
    education: [],
    highlights: [],
    teaching: [],
    researchInterests: []
  };

  return {
    ...profile,
    education: Array.isArray(profile.education) ? profile.education : [],
    highlights: Array.isArray(profile.highlights) ? profile.highlights : [],
    teaching: Array.isArray(profile.teaching) ? profile.teaching : [],
    researchInterests: Array.isArray(profile.researchInterests) ? profile.researchInterests : []
  };
}

export async function getAllPublications() {
  return readJsonFiles(publicationsDir).map(item => ensureSlug(item));
}

export async function getAllResearchDocuments() {
  return readJsonFiles(researchDir).map(item => ensureSlug(item));
}

export async function getInnovatrixProfile() {
  const items = readJsonFiles(innovatrixDir);
  const profile = items[0] || {
    name: 'Innovatrix',
    tagline: 'AI innovation studio',
    enabled: true,
    showOnHomepage: true,
    sections: []
  };

  return {
    ...profile,
    enabled: profile.enabled !== false,
    showOnHomepage: profile.showOnHomepage !== false,
    sections: Array.isArray(profile.sections) ? profile.sections.filter(section => section.enabled !== false) : []
  };
}
