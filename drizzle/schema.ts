import { index, int, json, mysqlEnum, mysqlTable, text, timestamp, varchar } from "drizzle-orm/mysql-core";

export const users = mysqlTable("users", {
  id: int("id").autoincrement().primaryKey(),
  openId: varchar("openId", { length: 64 }).notNull().unique(),
  name: text("name"),
  email: varchar("email", { length: 320 }),
  loginMethod: varchar("loginMethod", { length: 64 }),
  role: mysqlEnum("role", ["user", "admin"]).default("user").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
  lastSignedIn: timestamp("lastSignedIn").defaultNow().notNull(),
});

export const workerClients = mysqlTable("worker_clients", {
  id: varchar("id", { length: 36 }).primaryKey(),
  ownerId: int("ownerId").notNull().references(() => users.id, { onDelete: "cascade" }),
  displayName: varchar("displayName", { length: 96 }).notNull(),
  tokenHash: varchar("tokenHash", { length: 64 }).notNull().unique(),
  inventoryJson: json("inventoryJson").$type<LocalSourceFile[]>().notNull(),
  lastSeenAt: timestamp("lastSeenAt"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
}, table => [index("worker_owner_idx").on(table.ownerId)]);

export const videoJobs = mysqlTable("video_jobs", {
  id: varchar("id", { length: 36 }).primaryKey(),
  ownerId: int("ownerId").notNull().references(() => users.id, { onDelete: "cascade" }),
  workerId: varchar("workerId", { length: 36 }).notNull().references(() => workerClients.id, { onDelete: "cascade" }),
  sourceKey: varchar("sourceKey", { length: 256 }).notNull(),
  sourceName: varchar("sourceName", { length: 256 }).notNull(),
  status: mysqlEnum("status", ["queued", "claimed", "processing", "awaiting_review", "complete", "failed", "cancelled"]).default("queued").notNull(),
  stage: varchar("stage", { length: 64 }).default("queued").notNull(),
  progress: int("progress").default(0).notNull(),
  configJson: json("configJson").$type<VideoJobConfig>().notNull(),
  outputLocalPath: text("outputLocalPath"),
  errorMessage: text("errorMessage"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
  startedAt: timestamp("startedAt"),
  completedAt: timestamp("completedAt"),
}, table => [
  index("job_worker_status_idx").on(table.workerId, table.status),
  index("job_owner_created_idx").on(table.ownerId, table.createdAt),
]);

export const videoSegments = mysqlTable("video_segments", {
  id: varchar("id", { length: 36 }).primaryKey(),
  jobId: varchar("jobId", { length: 36 }).notNull().references(() => videoJobs.id, { onDelete: "cascade" }),
  position: int("position").notNull(),
  startMs: int("startMs").notNull(),
  endMs: int("endMs").notNull(),
  asrTextZh: text("asrTextZh"),
  ocrTextZh: text("ocrTextZh"),
  sourceTextZh: text("sourceTextZh"),
  translatedTextVi: text("translatedTextVi"),
  voicePath: text("voicePath"),
  voiceDurationMs: int("voiceDurationMs"),
  confidence: int("confidence").default(0).notNull(),
  ocrAuditJson: json("ocrAuditJson").$type<OcrAuditTrail>(),
  needsReview: int("needsReview").default(0).notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
}, table => [index("segment_job_position_idx").on(table.jobId, table.position)]);

export type LocalSourceFile = { key: string; name: string; sizeBytes: number; modifiedAtMs: number };
export type VideoJobConfig = {
  roi: { xPercent: number; yPercent: number; widthPercent: number; heightPercent: number; blurPx: number };
  targetLanguage: "vi";
  voice: { provider: "edge" | "piper"; name: string; maxTempo: number };
  audioMode: "replace" | "duck" | "keep";
  ocr: { enabled: boolean; sampleFrames: number; minConfidence: number; llmCorrection: boolean };
};
export type OcrAuditTrail = {
  originalText: string | null;
  ocrConfidence: number;
  frameAgreement: number;
  candidates: Array<{ text: string; hits: number; confidence: number }>;
  llmCorrection?: { attempted: boolean; model: string | null; correctedText: string; accepted: boolean; modelConfidence: number; rationale: string; needsReview: boolean };
};

export type User = typeof users.$inferSelect;
export type InsertUser = typeof users.$inferInsert;
export type WorkerClient = typeof workerClients.$inferSelect;
export type VideoJob = typeof videoJobs.$inferSelect;
export type VideoSegment = typeof videoSegments.$inferSelect;
