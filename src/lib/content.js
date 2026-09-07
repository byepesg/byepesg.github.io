import fs from 'fs';
import path from 'path';

const projectsDir = path.resolve('./src/content/projects');

export async function getAllProjects() {
  try {
    const files = fs.readdirSync(projectsDir).filter(f => f.endsWith('.json'));
    return files.map(f => JSON.parse(fs.readFileSync(path.join(projectsDir, f), 'utf-8')));
  } catch (e) {
    return [];
  }
}
