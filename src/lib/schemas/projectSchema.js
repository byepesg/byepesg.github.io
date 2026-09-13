import { z } from 'zod';

export const ProjectSchema = z.object({
  slug: z.string().min(1),
  title: z.string().min(1),
  year: z.number().int().optional(),
  featured: z.boolean().optional(),
  summary: z.string().optional(),
  categories: z.array(z.string()).optional(),
  links: z.record(z.string()).optional()
});

export function validateProject(data) {
  return ProjectSchema.parse(data);
}
