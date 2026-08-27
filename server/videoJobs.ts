import { and, asc, desc, eq } from "drizzle-orm";
import { createHash, randomUUID } from "node:crypto";
import { getDb } from "./db";
import { LocalSourceFile, OcrAuditTrail, VideoJobConfig, videoJobs, videoSegments, workerClients } from "../drizzle/schema";

export const hashWorkerToken = (token: string) => createHash("sha256").update(token).digest("hex");

export const defaultJobConfig: VideoJobConfig = {
  roi: { xPercent: 10, yPercent: 76, widthPercent: 80, heightPercent: 9, blurPx: 18 },
  targetLanguage: "vi",
  voice: { provider: "edge", name: "vi-VN-HoaiMyNeural", maxTempo: 1.15 },
  audioMode: "replace",
  ocr: { enabled: true, sampleFrames: 7, minConfidence: 72, llmCorrection: true },
};

export async function createWorkerPairing(ownerId: number, displayName: string, tokenHash: string) {
  const db = await getDb();
  if (!db) throw new Error("Database is unavailable");
  const worker = { id: randomUUID(), ownerId, displayName, tokenHash, inventoryJson: [] as LocalSourceFile[] };
  await db.insert(workerClients).values(worker);
  return worker;
}

export async function findWorkerByToken(tokenHash: string) {
  const db = await getDb();
  if (!db) throw new Error("Database is unavailable");
  const rows = await db.select().from(workerClients).where(eq(workerClients.tokenHash, tokenHash)).limit(1);
  return rows[0] ?? null;
}

export async function heartbeatWorker(workerId: string, inventory: LocalSourceFile[]) {
  const db = await getDb();
  if (!db) throw new Error("Database is unavailable");
  await db.update(workerClients).set({ inventoryJson: inventory, lastSeenAt: new Date() }).where(eq(workerClients.id, workerId));
}

export async function listWorkers(ownerId: number) {
  const db = await getDb();
  if (!db) return [];
  return db.select().from(workerClients).where(eq(workerClients.ownerId, ownerId)).orderBy(desc(workerClients.updatedAt));
}

export async function createVideoJob(input: { ownerId: number; workerId: string; sourceKey: string; sourceName: string; config: VideoJobConfig }) {
  const db = await getDb();
  if (!db) throw new Error("Database is unavailable");
  const worker = await db.select({ id: workerClients.id }).from(workerClients)
    .where(and(eq(workerClients.id, input.workerId), eq(workerClients.ownerId, input.ownerId))).limit(1);
  if (!worker[0]) throw new Error("Worker không thuộc về tài khoản hiện tại");
  const job = {
    id: randomUUID(),
    ownerId: input.ownerId,
    workerId: input.workerId,
    sourceKey: input.sourceKey,
    sourceName: input.sourceName,
    configJson: input.config,
    status: "queued" as const,
    stage: "queued",
    progress: 0,
  };
  await db.insert(videoJobs).values(job);
  return job;
}

export async function listVideoJobs(ownerId: number) {
  const db = await getDb();
  if (!db) return [];
  return db.select().from(videoJobs).where(eq(videoJobs.ownerId, ownerId)).orderBy(desc(videoJobs.createdAt));
}

export async function getVideoJobForOwner(jobId: string, ownerId: number) {
  const db = await getDb();
  if (!db) return null;
  const jobs = await db.select().from(videoJobs).where(and(eq(videoJobs.id, jobId), eq(videoJobs.ownerId, ownerId))).limit(1);
  const job = jobs[0];
  if (!job) return null;
  const segments = await db.select().from(videoSegments).where(eq(videoSegments.jobId, job.id)).orderBy(asc(videoSegments.position));
  return { ...job, segments };
}

export async function claimNextJob(workerId: string) {
  const db = await getDb();
  if (!db) throw new Error("Database is unavailable");
  const jobs = await db.select().from(videoJobs)
    .where(and(eq(videoJobs.workerId, workerId), eq(videoJobs.status, "queued")))
    .orderBy(asc(videoJobs.createdAt)).limit(1);
  const job = jobs[0];
  if (!job) return null;
  const segments = await db.select().from(videoSegments).where(eq(videoSegments.jobId, job.id)).orderBy(asc(videoSegments.position));
  await db.update(videoJobs).set({ status: "claimed", stage: "claimed", progress: 1, startedAt: new Date() }).where(and(eq(videoJobs.id, job.id), eq(videoJobs.status, "queued")));
  return { ...job, status: "claimed" as const, stage: "claimed", progress: 1, resumeStage: job.stage, segments };
}

export async function reportVideoJob(workerId: string, input: { jobId: string; status?: "claimed" | "processing" | "awaiting_review" | "complete" | "failed" | "cancelled"; stage: string; progress: number; outputLocalPath?: string; errorMessage?: string }) {
  const db = await getDb();
  if (!db) throw new Error("Database is unavailable");
  const values = {
    ...input,
    progress: Math.max(0, Math.min(100, Math.round(input.progress))),
    completedAt: input.status === "complete" || input.status === "failed" || input.status === "cancelled" ? new Date() : undefined,
  };
  await db.update(videoJobs).set(values).where(and(eq(videoJobs.id, input.jobId), eq(videoJobs.workerId, workerId)));
}

export async function replaceSegments(workerId: string, jobId: string, segments: Array<{ position: number; startMs: number; endMs: number; asrTextZh?: string; ocrTextZh?: string; sourceTextZh?: string; translatedTextVi?: string; voicePath?: string; voiceDurationMs?: number; confidence?: number; ocrAuditJson?: OcrAuditTrail; needsReview?: boolean }>) {
  const db = await getDb();
  if (!db) throw new Error("Database is unavailable");
  const job = await db.select().from(videoJobs).where(and(eq(videoJobs.id, jobId), eq(videoJobs.workerId, workerId))).limit(1);
  if (!job[0]) throw new Error("Job not found for this worker");
  await db.delete(videoSegments).where(eq(videoSegments.jobId, jobId));
  if (segments.length) {
    await db.insert(videoSegments).values(segments.map(segment => ({ id: randomUUID(), jobId, ...segment, needsReview: segment.needsReview ? 1 : 0 })));
  }
}

export async function saveReviewedSegments(ownerId: number, jobId: string, segments: Array<{ id: string; sourceTextZh?: string; translatedTextVi: string; startMs: number; endMs: number }>) {
  const db = await getDb();
  if (!db) throw new Error("Database is unavailable");
  const job = await db.select().from(videoJobs).where(and(eq(videoJobs.id, jobId), eq(videoJobs.ownerId, ownerId))).limit(1);
  if (!job[0]) throw new Error("Không tìm thấy job thuộc tài khoản hiện tại");
  for (const segment of segments) {
    await db.update(videoSegments).set({
      sourceTextZh: segment.sourceTextZh,
      translatedTextVi: segment.translatedTextVi,
      startMs: segment.startMs,
      endMs: segment.endMs,
      needsReview: 0,
    }).where(and(eq(videoSegments.id, segment.id), eq(videoSegments.jobId, jobId)));
  }
  await db.update(videoJobs).set({ status: "queued", stage: "render_from_review", progress: 80, errorMessage: null, completedAt: null }).where(eq(videoJobs.id, jobId));
}
