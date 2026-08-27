import { describe, expect, it } from "vitest";
import { defaultJobConfig, hashWorkerToken } from "./videoJobs";

describe("local worker contract", () => {
  it("hashes pairing tokens deterministically without retaining the raw token", () => {
    expect(hashWorkerToken("dubvi_worker_example")).toHaveLength(64);
    expect(hashWorkerToken("dubvi_worker_example")).toBe(hashWorkerToken("dubvi_worker_example"));
    expect(hashWorkerToken("dubvi_worker_example")).not.toBe(hashWorkerToken("other_token"));
  });

  it("ships a fixed bottom-center ROI configuration for the blur-first MVP", () => {
    expect(defaultJobConfig.roi).toEqual({ xPercent: 10, yPercent: 76, widthPercent: 80, heightPercent: 9, blurPx: 18 });
    expect(defaultJobConfig.voice.maxTempo).toBeLessThanOrEqual(1.15);
    expect(defaultJobConfig.ocr.llmCorrection).toBe(true);
  });
});
