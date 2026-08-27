import { COOKIE_NAME } from "@shared/const";
import { TRPCError } from "@trpc/server";
import { nanoid } from "nanoid";
import { z } from "zod";
import { getSessionCookieOptions } from "./_core/cookies";
import { systemRouter } from "./_core/systemRouter";
import { protectedProcedure, publicProcedure, router } from "./_core/trpc";
import {
  claimNextJob,
  createVideoJob,
  createWorkerPairing,
  defaultJobConfig,
  findWorkerByToken,
  getVideoJobForOwner,
  hashWorkerToken,
  heartbeatWorker,
  listVideoJobs,
  listWorkers,
  replaceSegments,
  reportVideoJob,
  saveReviewedSegments,
} from "./videoJobs";
import { correctOcrContext } from "./ocrCorrection";

const sourceFileSchema = z.object({ key: z.string().min(1).max(256), name: z.string().min(1).max(256), sizeBytes: z.number().int().nonnegative(), modifiedAtMs: z.number().int().nonnegative() });
const jobConfigSchema = z.object({
  roi: z.object({ xPercent: z.number().min(0).max(100), yPercent: z.number().min(0).max(100), widthPercent: z.number().min(1).max(100), heightPercent: z.number().min(1).max(100), blurPx: z.number().int().min(1).max(60) }),
  targetLanguage: z.literal("vi"),
  voice: z.object({ provider: z.enum(["edge", "piper"]), name: z.string().min(1), maxTempo: z.number().min(1).max(1.5) }),
  audioMode: z.enum(["replace", "duck", "keep"]),
  ocr: z.object({ enabled: z.boolean(), sampleFrames: z.number().int().min(1).max(30), minConfidence: z.number().int().min(0).max(100), llmCorrection: z.boolean() }),
});
const workerTokenSchema = z.object({ token: z.string().min(32).max(128) });
const ocrCandidateSchema = z.object({ text: z.string().min(1).max(2000), hits: z.number().int().positive(), confidence: z.number().int().min(0).max(100) });
const ocrLlmAuditSchema = z.object({ attempted: z.boolean(), model: z.string().nullable(), correctedText: z.string(), accepted: z.boolean(), modelConfidence: z.number().int().min(0).max(100), rationale: z.string(), needsReview: z.boolean() });
const ocrAuditSchema = z.object({ originalText: z.string().nullable(), ocrConfidence: z.number().int().min(0).max(100), frameAgreement: z.number().int().min(0).max(100), candidates: z.array(ocrCandidateSchema).max(30), llmCorrection: ocrLlmAuditSchema.optional() });
const segmentSchema = z.object({
  position: z.number().int().nonnegative(), startMs: z.number().int().nonnegative(), endMs: z.number().int().positive(),
  asrTextZh: z.string().optional(), ocrTextZh: z.string().optional(), sourceTextZh: z.string().optional(), translatedTextVi: z.string().optional(),
  voicePath: z.string().optional(), voiceDurationMs: z.number().int().nonnegative().optional(), confidence: z.number().int().min(0).max(100).optional(), ocrAuditJson: ocrAuditSchema.optional(), needsReview: z.boolean().optional(),
});
const reviewedSegmentSchema = z.object({
  id: z.string().uuid(), sourceTextZh: z.string().max(4000).optional(), translatedTextVi: z.string().min(1).max(4000),
  startMs: z.number().int().nonnegative(), endMs: z.number().int().positive(),
});
const ocrCorrectionSchema = z.object({
  ocrText: z.string().min(1).max(2000), ocrConfidence: z.number().int().min(0).max(100), frameAgreement: z.number().int().min(0).max(100),
  candidates: z.array(ocrCandidateSchema).max(30),
  asrTextZh: z.string().max(2000).optional(), asrConfidence: z.number().int().min(0).max(100).optional(),
});

async function requireWorker(token: string) {
  const worker = await findWorkerByToken(hashWorkerToken(token));
  if (!worker) throw new TRPCError({ code: "UNAUTHORIZED", message: "Worker token không hợp lệ" });
  return worker;
}

export const appRouter = router({
  system: systemRouter,
  auth: router({
    me: publicProcedure.query(opts => opts.ctx.user),
    logout: publicProcedure.mutation(({ ctx }) => {
      const cookieOptions = getSessionCookieOptions(ctx.req);
      ctx.res.clearCookie(COOKIE_NAME, { ...cookieOptions, maxAge: -1 });
      return { success: true } as const;
    }),
  }),
  jobs: router({
    list: protectedProcedure.query(({ ctx }) => listVideoJobs(ctx.user.id)),
    get: protectedProcedure.input(z.object({ jobId: z.string().uuid() })).query(({ ctx, input }) => getVideoJobForOwner(input.jobId, ctx.user.id)),
    create: protectedProcedure.input(z.object({ workerId: z.string().uuid(), sourceKey: z.string().min(1).max(256), sourceName: z.string().min(1).max(256), config: jobConfigSchema.default(defaultJobConfig) })).mutation(({ ctx, input }) => createVideoJob({ ownerId: ctx.user.id, ...input })),
    saveReviewedSegments: protectedProcedure.input(z.object({ jobId: z.string().uuid(), segments: z.array(reviewedSegmentSchema).min(1).max(500) })).mutation(async ({ ctx, input }) => {
      await saveReviewedSegments(ctx.user.id, input.jobId, input.segments);
      return { ok: true, count: input.segments.length };
    }),
  }),
  workers: router({
    list: protectedProcedure.query(({ ctx }) => listWorkers(ctx.user.id)),
    createPairing: protectedProcedure.input(z.object({ displayName: z.string().min(2).max(96) })).mutation(async ({ ctx, input }) => {
      const token = `dubvi_${nanoid(48)}`;
      const worker = await createWorkerPairing(ctx.user.id, input.displayName, hashWorkerToken(token));
      return { worker: { id: worker.id, displayName: worker.displayName }, token };
    }),
  }),
  worker: router({
    heartbeat: publicProcedure.input(workerTokenSchema.extend({ inventory: z.array(sourceFileSchema).max(500) })).mutation(async ({ input }) => {
      const worker = await requireWorker(input.token);
      await heartbeatWorker(worker.id, input.inventory);
      return { workerId: worker.id, accepted: input.inventory.length };
    }),
    claim: publicProcedure.input(workerTokenSchema).mutation(async ({ input }) => {
      const worker = await requireWorker(input.token);
      return claimNextJob(worker.id);
    }),
    report: publicProcedure.input(workerTokenSchema.extend({ jobId: z.string().uuid(), status: z.enum(["claimed", "processing", "awaiting_review", "complete", "failed", "cancelled"]).optional(), stage: z.string().min(1).max(64), progress: z.number().int().min(0).max(100), outputLocalPath: z.string().max(1024).optional(), errorMessage: z.string().max(8000).optional() })).mutation(async ({ input }) => {
      const worker = await requireWorker(input.token);
      await reportVideoJob(worker.id, input);
      return { ok: true };
    }),
    replaceSegments: publicProcedure.input(workerTokenSchema.extend({ jobId: z.string().uuid(), segments: z.array(segmentSchema).max(500) })).mutation(async ({ input }) => {
      const worker = await requireWorker(input.token);
      await replaceSegments(worker.id, input.jobId, input.segments);
      return { ok: true, count: input.segments.length };
    }),
    correctOcr: publicProcedure.input(workerTokenSchema.merge(ocrCorrectionSchema)).mutation(async ({ input }) => {
      await requireWorker(input.token);
      return correctOcrContext(input);
    }),
  }),
});

export type AppRouter = typeof appRouter;
