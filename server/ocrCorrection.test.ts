import { describe, expect, it } from "vitest";
import { shouldEscalateOcr } from "./ocrCorrection";

describe("OCR correction policy", () => {
  it("skips LLM when multi-frame OCR is highly confident and aligned", () => {
    expect(shouldEscalateOcr({ ocrText: "你说过两天来看我", ocrConfidence: 99, frameAgreement: 100, candidates: [] })).toBe(false);
  });

  it("escalates low confidence or conflicting frame evidence", () => {
    expect(shouldEscalateOcr({ ocrText: "你说过两天来着我", ocrConfidence: 82, frameAgreement: 71, candidates: [] })).toBe(true);
  });
});
