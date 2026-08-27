import { correctOcrContext } from "./ocrCorrection";

async function main() {
  const result = await correctOcrContext({
    ocrText: "你说过两天来着我",
    ocrConfidence: 82,
    frameAgreement: 72,
    candidates: [
      { text: "你说过两天来着我", hits: 4, confidence: 82 },
      { text: "你说过两天来看我", hits: 2, confidence: 79 },
    ],
    asrTextZh: "你说过两天来看我",
    asrConfidence: 91,
  });
  console.log(JSON.stringify(result, null, 2));
}

void main();
