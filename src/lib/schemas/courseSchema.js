import { z } from 'zod';

export const LectureSchema = z.object({
  number: z.number().int().optional(),
  title: z.string().optional(),
  date: z.string().optional(),
  links: z.record(z.string()).optional()
});

export const CourseSchema = z.object({
  slug: z.string().min(1),
  title: z.string().min(1),
  semester: z.string().optional(),
  active: z.boolean().optional(),
  featured: z.boolean().optional(),
  summary: z.string().optional(),
  lectures: z.array(LectureSchema).optional()
});

export function validateCourse(data) {
  return CourseSchema.parse(data);
}
