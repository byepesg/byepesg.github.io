import fs from 'fs';
import path from 'path';
import { validateProject } from './schemas/projectSchema.js';
import { validateCourse } from './schemas/courseSchema.js';

const projectsDir = path.resolve('./src/content/projects');
const coursesDir = path.resolve('./src/content/courses');

function loadJsonFiles(dir) {
  try {
    const files = fs.readdirSync(dir).filter(f => f.endsWith('.json'));
    return files.map(f => JSON.parse(fs.readFileSync(path.join(dir, f), 'utf-8')));
  } catch (e) {
    return [];
  }
}

export async function getAllProjects() {
  const data = loadJsonFiles(projectsDir);
  // validate
  return data.map(d => validateProject(d));
}

export async function getAllCourses() {
  const data = loadJsonFiles(coursesDir);
  return data.map(d => validateCourse(d));
}
