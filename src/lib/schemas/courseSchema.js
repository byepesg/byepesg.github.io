import { z } from 'zod';

export const LectureSchema = z.object({
  number: z.number().int().optional(),
  title: z.string().optional(),
  date: z.string().optional(),
  links: z.record(z.string()).optional()
});

export const CourseResourceSchema = z.object({
  label: z.string().min(1),
  href: z.union([
    z.string().url(),
    z.string().regex(/^\/[A-Za-z0-9._~!$&'()*+,;=:@%\/-]*$/)
  ]),
  type: z.enum(['folder', 'drive', 'document', 'link']).optional(),
  private: z.boolean().optional().default(true),
  note: z.string().optional()
});

export const CourseSchema = z.object({
  slug: z.string().min(1),
  title: z.string().min(1),
  semester: z.string().optional(),
  active: z.boolean().optional(),
  featured: z.boolean().optional(),
  summary: z.string().optional(),
  lectures: z.array(LectureSchema).optional(),
  resources: z.array(CourseResourceSchema).optional()
});

export function validateCourse(data) {
  return CourseSchema.parse(data);
}
