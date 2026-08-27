import { invokeLLM } from "./_core/llm";

export type OcrCandidate = { text: string; hits: number; confidence: number };
export type OcrCorrectionInput = {
  ocrText: string;
  ocrConfidence: number;
  frameAgreement: number;
  candidates: OcrCandidate[];
  asrTextZh?: string;
  asrConfidence?: number;
};
export type OcrCorrectionResult = {
  attempted: boolean;
  model: string | null;
  correctedText: string;
  accepted: boolean;
  modelConfidence: number;
  rationale: string;
  needsReview: boolean;
};

const normalizeChinese = (value: string) => value.replace(/[\s.,!?，。！？、；：;:'"“”‘’（）()\-—_]/g, "");

export function shouldEscalateOcr(input: OcrCorrectionInput) {
  if (!input.ocrText.trim()) return false;
  if (input.ocrConfidence < 96 || input.frameAgreement < 86) return true;
  const asr = normalizeChinese(input.asrTextZh ?? "");
  const ocr = normalizeChinese(input.ocrText);
  return Boolean(asr && input.asrConfidence && input.asrConfidence >= 65 && asr !== ocr);
}

function safeCorrection(original: string, result: OcrCorrectionResult): OcrCorrectionResult {
  const value = result.correctedText.trim();
  const originalCompact = normalizeChinese(original);
  const correctedCompact = normalizeChinese(value);
  const plausibleLength = correctedCompact.length >= Math.max(1, Math.floor(originalCompact.length * 0.45))
    && correctedCompact.length <= Math.max(2, Math.ceil(originalCompact.length * 1.8));
  const accepted = result.modelConfidence >= 85 && value !== original && plausibleLength;
  return { ...result, correctedText: accepted ? value : original, accepted, needsReview: accepted || result.needsReview };
}

export async function correctOcrContext(input: OcrCorrectionInput): Promise<OcrCorrectionResult> {
  if (!shouldEscalateOcr(input)) {
    return { attempted: false, model: null, correctedText: input.ocrText, accepted: false, modelConfidence: 100, rationale: "OCR đa khung hình đã đủ nhất quán; không gọi LLM.", needsReview: false };
  }
  const model = "claude-haiku-4-5";
  try {
    const result = await invokeLLM({
      model,
      maxTokens: 220,
      messages: [
        { role: "system", content: "Bạn là bộ sửa lỗi OCR tiếng Trung giản thể. Chỉ sửa lỗi nhận diện rõ ràng dựa trên nhiều frame và ASR. Tuyệt đối không dịch, diễn giải, thêm nội dung, thay đổi tên riêng hoặc đoán khi thiếu bằng chứng. Nếu không chắc, giữ nguyên OCR." },
        { role: "user", content: JSON.stringify(input) },
      ],
      outputSchema: {
        name: "ocr_context_correction",
        strict: true,
        schema: {
          type: "object",
          properties: {
            correctedText: { type: "string" },
            modelConfidence: { type: "integer", minimum: 0, maximum: 100 },
            rationale: { type: "string" },
            needsReview: { type: "boolean" },
          },
          required: ["correctedText", "modelConfidence", "rationale", "needsReview"],
          additionalProperties: false,
        },
      },
    });
    const content = result.choices[0]?.message.content;
    const contentText = typeof content === "string"
      ? content
      : Array.isArray(content)
        ? content.filter(part => part.type === "text").map(part => part.text).join("")
        : "";
    const parsed = JSON.parse(contentText || "{}") as Omit<OcrCorrectionResult, "attempted" | "model" | "accepted">;
    if (typeof parsed.correctedText !== "string" || typeof parsed.modelConfidence !== "number" || typeof parsed.rationale !== "string" || typeof parsed.needsReview !== "boolean") {
      throw new Error(`LLM trả JSON correction không đúng schema: ${contentText.slice(0, 500) || JSON.stringify(content ?? "non-text content").slice(0, 500)}`);
    }
    return safeCorrection(input.ocrText, { attempted: true, model, accepted: false, ...parsed });
  } catch (error) {
    return { attempted: true, model, correctedText: input.ocrText, accepted: false, modelConfidence: 0, rationale: `LLM correction không khả dụng: ${error instanceof Error ? error.message : "unknown error"}`, needsReview: true };
  }
}
