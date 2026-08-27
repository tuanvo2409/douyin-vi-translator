/**
 * STYLE REMINDER — Bàn dựng phim biên tập:
 * Swiss editorial workspace, deep ink-navy canvas, ivory inspection panels,
 * Signal Amber only for intentional playback/render actions. The vertical preview is the visual center.
 */
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { useAuth } from "@/_core/hooks/useAuth";
import { trpc } from "@/lib/trpc";
import { startLogin } from "@/const";
import { toast } from "sonner";
import {
  ArrowLeft,
  AudioLines,
  Check,
  ChevronDown,
  ChevronRight,
  Clock3,
  Film,
  Languages,
  MoreHorizontal,
  Pause,
  Play,
  Plus,
  SlidersHorizontal,
  Sparkles,
  Upload,
  Volume2,
  WandSparkles,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

type Segment = {
  id: string;
  start: number;
  end: number;
  zh: string;
  vi: string;
  voice: string;
  fit: "Khớp" | "Rút 0.12s" | "Cần duyệt";
};

const segments: Segment[] = [
  {
    id: "s1",
    start: 3.0,
    end: 4.1,
    zh: "你说过两天来看我",
    vi: "Em từng nói hai ngày nữa sẽ đến",
    voice: "1.04s / slot 1.10s",
    fit: "Khớp",
  },
  {
    id: "s2",
    start: 4.1,
    end: 5.3,
    zh: "一等就是一年多",
    vi: "Đợi một lần đã hơn một năm",
    voice: "1.13s / slot 1.20s",
    fit: "Khớp",
  },
  {
    id: "s3",
    start: 5.3,
    end: 6.8,
    zh: "风吹过那条街",
    vi: "Gió lướt qua con phố ấy",
    voice: "1.42s / slot 1.50s",
    fit: "Rút 0.12s",
  },
];

const formatTime = (seconds: number) =>
  `00:${seconds.toFixed(2).padStart(5, "0")}`;

export default function Home() {
  const { user, isAuthenticated, loading: authLoading } = useAuth();
  const [activeId, setActiveId] = useState("s1");
  const [isPlaying, setIsPlaying] = useState(false);
  const [blur, setBlur] = useState([18]);
  const [rendering, setRendering] = useState(false);
  const [pairingToken, setPairingToken] = useState<string | null>(null);
  const [reviewText, setReviewText] = useState<Record<string, string>>({});
  const [reviewZh, setReviewZh] = useState<Record<string, string>>({});
  const workerQuery = trpc.workers.list.useQuery(undefined, { enabled: isAuthenticated, refetchInterval: 15_000 });
  const jobQuery = trpc.jobs.list.useQuery(undefined, { enabled: isAuthenticated, refetchInterval: 8_000 });
  const worker = workerQuery.data?.[0];
  const reviewJob = jobQuery.data?.find(job => job.status === "awaiting_review") ?? null;
  const reviewDetailQuery = trpc.jobs.get.useQuery(
    { jobId: reviewJob?.id ?? "00000000-0000-0000-0000-000000000000" },
    { enabled: Boolean(isAuthenticated && reviewJob?.id) },
  );
  const pairingMutation = trpc.workers.createPairing.useMutation({
    onSuccess: data => {
      setPairingToken(data.token);
      toast.success("Worker pairing token đã tạo", { description: "Chép token vào file .env của local worker rồi chạy lệnh heartbeat/watch." });
    },
    onError: error => toast.error("Không tạo được pairing token", { description: error.message }),
  });
  const createJobMutation = trpc.jobs.create.useMutation({
    onSuccess: job => {
      setRendering(false);
      void jobQuery.refetch();
      toast.success("Đã xếp hàng render local", { description: `Job ${job.id.slice(0, 8)} đang chờ worker claim.` });
    },
    onError: error => {
      setRendering(false);
      toast.error("Không tạo được job", { description: error.message });
    },
  });
  const saveReviewMutation = trpc.jobs.saveReviewedSegments.useMutation({
    onSuccess: () => {
      void jobQuery.refetch();
      toast.success("Đã lưu segment và xếp lại worker", { description: "Worker sẽ render lại từ voice file local đã tạo trước đó." });
    },
    onError: error => toast.error("Không lưu được segment", { description: error.message }),
  });
  const activeSegment = useMemo(
    () => segments.find((segment) => segment.id === activeId) ?? segments[0],
    [activeId],
  );

  useEffect(() => {
    if (!reviewDetailQuery.data?.segments) return;
    setReviewText(Object.fromEntries(reviewDetailQuery.data.segments.map(segment => [segment.id, segment.translatedTextVi ?? ""])));
    setReviewZh(Object.fromEntries(reviewDetailQuery.data.segments.map(segment => [segment.id, segment.sourceTextZh ?? segment.ocrTextZh ?? segment.asrTextZh ?? ""])));
  }, [reviewDetailQuery.data?.id, reviewDetailQuery.data?.updatedAt]);

  const createPairing = () => {
    if (!isAuthenticated) {
      startLogin();
      return;
    }
    pairingMutation.mutate({ displayName: "Máy local của tôi" });
  };

  const handleRender = () => {
    if (!isAuthenticated) {
      toast("Đăng nhập để tạo job local", { description: "Pairing token sẽ chỉ hiện cho tài khoản của bạn." });
      startLogin();
      return;
    }
    if (!worker) {
      toast("Chưa thấy local worker", { description: "Tạo pairing token, chạy worker heartbeat rồi quay lại render." });
      return;
    }
    const firstLocalVideo = worker.inventoryJson?.[0];
    if (!firstLocalVideo) {
      toast("Worker chưa gửi video local", { description: "Đặt MP4 vào DUBVI_MEDIA_DIR rồi chạy `python dubvi_worker.py heartbeat`." });
      return;
    }
    setRendering(true);
    createJobMutation.mutate({
      workerId: worker.id,
      sourceKey: firstLocalVideo.key,
      sourceName: firstLocalVideo.name,
      config: {
        roi: { xPercent: 10, yPercent: 76, widthPercent: 80, heightPercent: 9, blurPx: blur[0] },
        targetLanguage: "vi",
        voice: { provider: "edge", name: "vi-VN-HoaiMyNeural", maxTempo: 1.15 },
        audioMode: "replace",
        ocr: { enabled: true, sampleFrames: 7, minConfidence: 72, llmCorrection: true },
      },
    });
  };

  const selectSegment = (segment: Segment) => {
    setActiveId(segment.id);
    setIsPlaying(false);
  };

  const saveReview = () => {
    if (!reviewDetailQuery.data) return;
    saveReviewMutation.mutate({
      jobId: reviewDetailQuery.data.id,
      segments: reviewDetailQuery.data.segments.map(segment => ({
        id: segment.id,
        sourceTextZh: reviewZh[segment.id]?.trim() || (segment.sourceTextZh ?? segment.ocrTextZh ?? segment.asrTextZh ?? undefined),
        translatedTextVi: reviewText[segment.id]?.trim() || segment.translatedTextVi || "[Cần nhập lời Việt]",
        startMs: segment.startMs,
        endMs: segment.endMs,
      })),
    });
  };

  return (
    <div className="studio-shell min-h-screen bg-[#10161d] text-[#f8f1e5]">
      <header className="studio-header">
        <div className="flex items-center gap-3">
          <button
            aria-label="Quay lại danh sách project"
            className="icon-button"
            onClick={() => toast("Danh sách project sẽ xuất hiện ở bản sau")}
          >
            <ArrowLeft size={18} />
          </button>
          <div className="brand-lockup">
            <img
              src="/manus-storage/douyin-vi-logo_7412455a.png"
              alt="Biểu tượng Douyin sang Việt"
              className="brand-mark"
            />
            <div>
              <p className="brand-name">DUBVI</p>
              <p className="brand-subtitle">LOCAL VIDEO TRANSLATOR</p>
            </div>
          </div>
        </div>

          <div className="hidden items-center gap-5 lg:flex">
            <div className="header-status">
              <span className="status-dot" />
              {worker ? `Worker: ${worker.displayName}` : "Chưa ghép worker"}
            </div>
          <button
            className="header-link"
            onClick={() => toast("Hệ thống chỉ lưu dự án trong phiên làm việc này")}
          >
            Quyền riêng tư
          </button>
          <button className="avatar-chip" onClick={isAuthenticated ? () => toast(`Đã đăng nhập: ${user?.name ?? "DUBVI user"}`) : startLogin}>{authLoading ? "…" : (user?.name?.slice(0, 2).toUpperCase() ?? "DN")}</button>
        </div>
      </header>

      <main className="studio-main">
        <aside className="project-rail">
          <div>
            <p className="eyebrow">PROJECT 01</p>
            <h1 className="project-title">Nét cũ<br />phố quen</h1>
            <div className="source-file">
              <div className="source-icon"><Film size={17} /></div>
              <div className="min-w-0">
                <p className="truncate text-[12px] font-semibold text-[#f8f1e5]">SnapTikTok…9921.mp4</p>
                <p className="mt-1 text-[10px] font-medium tracking-[0.08em] text-[#8b969d]">9:16 · 00:12.4 · 1080P</p>
              </div>
              <MoreHorizontal size={16} className="shrink-0 text-[#8b969d]" />
            </div>
          </div>

          <nav className="mode-nav" aria-label="Các bước xử lý video">
            <button className="mode-item" onClick={() => toast("Nguồn video đã được chọn")}> <Upload size={16} /> Nguồn <ChevronRight size={14} /> </button>
            <button className="mode-item is-active" onClick={() => toast("Đang chỉnh fixed blur box")}> <SlidersHorizontal size={16} /> Blur box <ChevronRight size={14} /> </button>
            <button className="mode-item" onClick={() => toast("Có 3 segment cần rà soát")}> <Languages size={16} /> Song ngữ <span className="mode-count">03</span> </button>
            <button className="mode-item" onClick={() => toast("Voice sẽ được fit theo slot từng câu")}> <AudioLines size={16} /> Voice fit <ChevronRight size={14} /> </button>
          </nav>

          <div className="rail-note">
            <div className="note-line"><span>PHƯƠNG ÁN</span><b>Fixed blur</b></div>
            <div className="note-line"><span>WORKER</span><b>{worker ? "Đã ghép" : "Chưa ghép"}</b></div>
            <div className="note-line"><span>JOB LOCAL</span><b>{jobQuery.data?.length ?? 0}</b></div>
            {pairingToken ? (
              <div className="pair-token">
                <span>PAIRING TOKEN</span>
                <code>{pairingToken}</code>
                <button onClick={() => {
                  void navigator.clipboard?.writeText(pairingToken);
                  toast.success("Đã chép token vào clipboard");
                }}>Copy token</button>
              </div>
            ) : (
              <button className="pair-button" onClick={createPairing} disabled={pairingMutation.isPending}>
                {pairingMutation.isPending ? "Đang tạo…" : "Ghép local worker"}
              </button>
            )}
          </div>
        </aside>

        <section className="editor-stage">
          <div className="stage-toolbar">
            <div className="flex items-center gap-3">
              <p className="eyebrow">PREVIEW / 9:16</p>
              <span className="thin-divider" />
              <span className="text-[11px] font-medium text-[#9ba6aa]">Blur layer 01</span>
            </div>
            <button className="compact-control" onClick={() => toast("Tỉ lệ preview đã khóa ở 9:16")}>Fit <ChevronDown size={14} /></button>
          </div>

          <div className="preview-area">
            <div className="preview-ruler left-ruler">
              <span>0</span><span>360</span><span>720</span><span>1080</span><span>1440</span><span>1920</span>
            </div>
            <div className="video-frame" aria-label="Video preview với lớp blur cố định">
              <img
                src="/manus-storage/video-preview-cafe_a579a458.jpg"
                alt="Preview video dọc trong nhà hàng"
                className="video-image"
              />
              <div className="video-scrim" />
              <div className="preview-corner top-left" />
              <div className="preview-corner top-right" />
              <div className="preview-corner bottom-left" />
              <div className="preview-corner bottom-right" />

              <div
                className="blur-region"
                style={{
                  backdropFilter: `blur(${blur[0]}px) saturate(0.72)`,
                  WebkitBackdropFilter: `blur(${blur[0]}px) saturate(0.72)`,
                }}
              >
                <span className="region-label">FIXED REGION · 01</span>
                <div className="subtitle-copy">
                  <p className="original-text">{activeSegment.zh}</p>
                  <p className="translated-text">{activeSegment.vi}</p>
                </div>
                <span className="resize-dot dot-top-left" />
                <span className="resize-dot dot-top-right" />
                <span className="resize-dot dot-bottom-left" />
                <span className="resize-dot dot-bottom-right" />
              </div>

              <div className="video-meta">
                <span>{formatTime(activeSegment.start)}</span>
                <span className="meta-pip" />
                <span>SEGMENT 0{segments.indexOf(activeSegment) + 1}</span>
              </div>
              <button
                aria-label={isPlaying ? "Dừng preview" : "Phát preview"}
                className="play-control"
                onClick={() => setIsPlaying((playing) => !playing)}
              >
                {isPlaying ? <Pause size={19} fill="currentColor" /> : <Play size={19} fill="currentColor" />}
              </button>
            </div>
            <div className="preview-ruler right-ruler">
              <span>1080</span><span>720</span><span>360</span><span>0</span>
            </div>
          </div>

          <div className="stage-caption">
            <span><Sparkles size={14} /> Blur giữ theo frame, nền vẫn chuyển động</span>
            <span className="hidden sm:inline">ROI 80% × 9% · Y = 76%</span>
          </div>
        </section>

        <aside className="inspector-panel">
          <div className="inspector-heading">
            <div>
              <p className="eyebrow">INSPECTOR</p>
              <h2>Blur region</h2>
            </div>
            <button className="icon-button small" onClick={() => toast("Mở thêm tùy chọn region")}> <MoreHorizontal size={17} /> </button>
          </div>

          <div className="inspector-section region-card">
            <div className="region-card-top">
              <span className="region-index">01</span>
              <div>
                <p className="text-[12px] font-bold text-[#17212a]">Bottom-center subtitle</p>
                <p className="mt-1 text-[10px] font-semibold uppercase tracking-[0.1em] text-[#778088]">Active 00:03.00 → 00:06.80</p>
              </div>
              <Check size={16} className="ml-auto text-[#258e78]" />
            </div>
          </div>

          <div className="inspector-section control-stack">
            <div className="control-heading"><span>Blur strength</span><b>{blur[0]}px</b></div>
            <Slider
              value={blur}
              min={4}
              max={30}
              step={1}
              onValueChange={setBlur}
              aria-label="Cường độ làm mờ"
              className="amber-slider"
            />
            <div className="slider-extents"><span>Mềm</span><span>Che rõ</span></div>
          </div>

          <div className="inspector-section">
            <div className="control-heading"><span>Frame geometry</span><button className="text-[#c77808]" onClick={() => toast("Đã khóa hình học theo box subtitle")}>Reset</button></div>
            <div className="geometry-grid">
              <div><span>X</span><b>10%</b></div><div><span>Y</span><b>76%</b></div>
              <div><span>W</span><b>80%</b></div><div><span>H</span><b>09%</b></div>
            </div>
          </div>

          <div className="inspector-section voice-preview">
            <div className="control-heading"><span>Voice slot</span><Volume2 size={15} /></div>
            <p className="voice-lang">vi-VN · Hoài My</p>
            <div className="voice-pulse" />
            <div className="voice-status"><Clock3 size={13} /> {activeSegment.voice}</div>
          </div>

          <Button className="w-full render-button" onClick={handleRender} disabled={rendering}>
            {rendering ? <span className="render-spinner" /> : <WandSparkles size={16} />}
            {rendering ? "Đang dựng preview" : "Render preview"}
          </Button>
          <p className="render-note">Video không upload. Render được xếp hàng cho worker chạy trên máy bạn.</p>
        </aside>
      </main>

      <section className="timeline-dock">
        <div className="timeline-topline">
          <div className="flex items-center gap-3">
            <p className="eyebrow">EDIT TIMELINE</p>
            <span className="timeline-duration">00:12.40</span>
          </div>
          <div className="timeline-actions">
            <button onClick={() => toast("Một segment trống đã được thêm vào timeline")}> <Plus size={15} /> Segment</button>
            <button onClick={() => toast("Auto-fit đã rà soát 3 slot voice")}> <Sparkles size={15} /> Auto-fit voice</button>
          </div>
        </div>

        <div className="ruler-row">
          {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].map((second) => (
            <span key={second} style={{ left: `${(second / 12) * 100}%` }}>{second}s</span>
          ))}
        </div>

        <div className="tracks-wrap">
          <div className="track-labels">
            <span><Film size={14} /> Video</span>
            <span><SlidersHorizontal size={14} /> Blur box</span>
            <span><Languages size={14} /> Việt ngữ</span>
            <span><AudioLines size={14} /> Voice</span>
          </div>
          <div className="tracks">
            <div className="track video-track"><div className="film-strip" /></div>
            <div className="track blur-track"><div className="blur-segment" style={{ left: "25%", width: "33%" }}>Fixed region · 01</div></div>
            <div className="track text-track">
              {segments.map((segment) => {
                const left = `${(segment.start / 12) * 100}%`;
                const width = `${((segment.end - segment.start) / 12) * 100}%`;
                return (
                  <button
                    key={segment.id}
                    className={`text-segment ${activeId === segment.id ? "is-selected" : ""}`}
                    style={{ left, width }}
                    onClick={() => selectSegment(segment)}
                  >
                    <span>{segment.vi}</span>
                  </button>
                );
              })}
            </div>
            <div className="track voice-track">
              {segments.map((segment, index) => {
                const left = `${(segment.start / 12) * 100}%`;
                const width = `${((segment.end - segment.start) / 12) * 100}%`;
                return <div key={segment.id} className={`wave-segment wave-${index + 1}`} style={{ left, width }} />;
              })}
            </div>
            <div className="playhead" style={{ left: `${(activeSegment.start / 12) * 100}%` }}><span /></div>
          </div>
        </div>
      </section>

      <section className="segment-dock">
        <div className="segment-dock-head">
          <div>
            <p className="eyebrow">TRANSCRIPT STRIPS</p>
            <h2>Giữ nhịp gốc. Đặt lời Việt vào đúng chỗ.</h2>
          </div>
          <div className="legend"><span className="legend-amber" /> Đang chọn <span className="legend-mint" /> Voice khớp</div>
        </div>
        <div className="segment-list">
          {segments.map((segment, index) => (
            <button
              className={`segment-strip ${activeId === segment.id ? "is-active" : ""}`}
              key={segment.id}
              onClick={() => selectSegment(segment)}
            >
              <span className="segment-number">0{index + 1}</span>
              <span className="segment-time">{formatTime(segment.start)}<br />{formatTime(segment.end)}</span>
              <span className="segment-zh">{segment.zh}</span>
              <span className="segment-vi">{segment.vi}</span>
              <span className="segment-voice"><Volume2 size={14} /> {segment.voice}</span>
              <span className={`fit-badge ${segment.fit === "Khớp" ? "is-good" : "is-warn"}`}>{segment.fit}</span>
            </button>
          ))}
        </div>
        {reviewDetailQuery.data ? (
          <div className="mt-7 border-t border-[#17212a]/15 pt-5">
            <div className="flex items-end justify-between gap-4">
              <div>
                <p className="eyebrow text-[#778086]">REVIEW QUEUE</p>
                <h3 className="mt-1 font-['Space_Grotesk'] text-[18px] font-semibold tracking-[-0.05em]">Voice chưa vừa nhịp — chỉnh lời Việt trước khi render lại.</h3>
              </div>
              <Button className="shrink-0 rounded-none bg-[#17212a] text-[#fff9ed] hover:bg-[#2c3941]" onClick={saveReview} disabled={saveReviewMutation.isPending}>
                {saveReviewMutation.isPending ? "Đang lưu…" : "Lưu & render lại"}
              </Button>
            </div>
            <div className="mt-4 grid gap-3">
              {reviewDetailQuery.data.segments.map((segment, index) => (
                <div key={segment.id} className="grid gap-3 border border-[#17212a]/15 bg-[#fff9ed] p-4 md:grid-cols-[92px_minmax(0,1fr)_minmax(0,1.2fr)]">
                  <div>
                    <p className="font-['Space_Grotesk'] text-[10px] font-bold text-[#7b8587]">SEGMENT 0{index + 1}</p>
                    <p className="mt-1 font-['Space_Grotesk'] text-[10px] text-[#59636b]">{formatTime(segment.startMs / 1000)} → {formatTime(segment.endMs / 1000)}</p>
                  </div>
                  <div>
                    <p className="text-[10px] font-bold uppercase tracking-[0.09em] text-[#7b8587]">Trung OCR · có thể sửa</p>
                    <textarea
                      className="mt-2 min-h-[54px] w-full resize-y border border-[#17212a]/15 bg-white p-2 text-[13px] font-semibold text-[#3d494d] outline-none transition focus:border-[#f5a524]"
                      value={reviewZh[segment.id] ?? ""}
                      onChange={event => setReviewZh(current => ({ ...current, [segment.id]: event.target.value }))}
                    />
                    {segment.ocrAuditJson ? (
                      <div className="mt-2 border-l-2 border-[#f5a524] pl-2 text-[10px] leading-[1.45] text-[#667177]">
                        <b>OCR gốc:</b> {segment.ocrAuditJson.originalText ?? "—"} · {segment.ocrAuditJson.ocrConfidence}% · agreement {segment.ocrAuditJson.frameAgreement}%
                        <br /><b>ASR:</b> {segment.asrTextZh ?? "—"}
                        {segment.ocrAuditJson.llmCorrection ? <>
                          <br /><b>LLM đề xuất:</b> {segment.ocrAuditJson.llmCorrection.correctedText} · {segment.ocrAuditJson.llmCorrection.modelConfidence}% · {segment.ocrAuditJson.llmCorrection.rationale}
                          <div className="mt-2 flex flex-wrap gap-2">
                            <button type="button" className="border border-[#17212a]/20 bg-white px-2 py-1 text-[10px] font-bold text-[#59636b] hover:border-[#f5a524]" onClick={() => setReviewZh(current => ({ ...current, [segment.id]: segment.ocrAuditJson?.originalText ?? "" }))}>Dùng OCR gốc</button>
                            <button type="button" className="border border-[#f5a524] bg-[#fff5dc] px-2 py-1 text-[10px] font-bold text-[#9a610b] hover:bg-[#ffe9b7]" onClick={() => setReviewZh(current => ({ ...current, [segment.id]: segment.ocrAuditJson?.llmCorrection?.correctedText ?? "" }))}>Áp dụng LLM</button>
                          </div>
                        </> : null}
                      </div>
                    ) : null}
                  </div>
                  <label className="block">
                    <span className="text-[10px] font-bold uppercase tracking-[0.09em] text-[#9a610b]">Lời Việt · có thể sửa</span>
                    <textarea
                      className="mt-2 min-h-[54px] w-full resize-y border border-[#17212a]/15 bg-white p-2 text-[13px] font-semibold text-[#17212a] outline-none transition focus:border-[#f5a524]"
                      value={reviewText[segment.id] ?? ""}
                      onChange={event => setReviewText(current => ({ ...current, [segment.id]: event.target.value }))}
                    />
                  </label>
                </div>
              ))}
            </div>
          </div>
        ) : reviewJob ? (
          <p className="mt-6 text-[12px] font-semibold text-[#7b8587]">Đang tải segment cần duyệt…</p>
        ) : null}
      </section>

      <footer className="studio-footer">
        <span>Fixed ROI · Multi-frame OCR · Duration-aware voice</span>
        <span>Prototype / v0.1</span>
      </footer>
    </div>
  );
}
