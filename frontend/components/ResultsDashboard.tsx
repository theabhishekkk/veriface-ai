'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  ShieldAlert,
  ShieldCheck,
  Zap,
  Layers,
  Sparkles,
  Info,
  RotateCcw,
  Flame,
  Grid
} from 'lucide-react';

export interface PredictResult {
  id?: string;
  filename: string;
  label: string; // 'Synthetic' or 'Real'
  is_deepfake: boolean;
  confidence: number;
  confidence_percentage: number;
  attention_heatmap: string;
  raw_heatmap?: string;
  execution_time_ms: number;
  model_name: string;
  mode: string;
  patch_analysis: {
    patch_count?: number;
    grid_size?: string;
    highest_activation_zone?: string;
    laplacian_sharpness?: number;
  };
  created_at?: string;
}

interface ResultsDashboardProps {
  originalImageUrl: string;
  result: PredictResult;
  onReset: () => void;
}

export const ResultsDashboard: React.FC<ResultsDashboardProps> = ({
  originalImageUrl,
  result,
  onReset,
}) => {
  // View mode: 'overlay' | 'side-by-side' | 'raw-heatmap'
  const [viewMode, setViewMode] = useState<'overlay' | 'side-by-side' | 'raw-heatmap'>('overlay');
  const [overlayAlpha, setOverlayAlpha] = useState<number>(0.52);

  const isSynthetic = result.is_deepfake || result.label.toLowerCase() === 'synthetic';
  const confidencePct = result.confidence_percentage || Math.round(result.confidence * 100);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4 }}
      className="w-full bg-slate-900/80 border border-slate-800 rounded-3xl p-6 sm:p-8 backdrop-blur-xl shadow-2xl space-y-8"
    >
      {/* Header Banner: Classification Badge & Latency */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-slate-800">
        <div>
          <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">
            Inference Verdict
          </span>
          <div className="flex items-center gap-3 mt-1.5">
            {isSynthetic ? (
              <div className="flex items-center gap-2.5 px-4 py-2 bg-red-950/80 border border-red-500/60 rounded-2xl glow-synthetic">
                <ShieldAlert className="w-6 h-6 text-red-400 animate-pulse" />
                <div>
                  <h3 className="text-xl font-bold text-red-200 leading-tight">
                    SYNTHETIC / DEEPFAKE
                  </h3>
                  <p className="text-xs text-red-300/80">
                    High likelihood of generative AI manipulation
                  </p>
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-2.5 px-4 py-2 bg-emerald-950/80 border border-emerald-500/60 rounded-2xl glow-real">
                <ShieldCheck className="w-6 h-6 text-emerald-400" />
                <div>
                  <h3 className="text-xl font-bold text-emerald-200 leading-tight">
                    AUTHENTIC MEDIA
                  </h3>
                  <p className="text-xs text-emerald-300/80">
                    No anomalous ViT attention artifacts detected
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center gap-3 self-start sm:self-auto">
          <div className="px-3.5 py-1.5 rounded-xl bg-slate-800/80 border border-slate-700/60 flex items-center gap-2 text-xs text-slate-300">
            <Zap className="w-3.5 h-3.5 text-amber-400" />
            <span>{result.execution_time_ms} ms</span>
          </div>
          <button
            onClick={onReset}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-medium rounded-xl border border-slate-700 flex items-center gap-2 transition active:scale-95"
          >
            <RotateCcw className="w-3.5 h-3.5 text-indigo-400" />
            <span>New Scan</span>
          </button>
        </div>
      </div>

      {/* Confidence Score Bar */}
      <div className="bg-slate-950/60 border border-slate-800/80 rounded-2xl p-5 space-y-2.5">
        <div className="flex items-center justify-between text-sm">
          <span className="font-medium text-slate-300 flex items-center gap-1.5">
            <Sparkles className="w-4 h-4 text-indigo-400" />
            ViT Model Confidence
          </span>
          <span
            className={`text-lg font-bold ${
              isSynthetic ? 'text-red-400' : 'text-emerald-400'
            }`}
          >
            {confidencePct.toFixed(1)}%
          </span>
        </div>

        {/* Animated Bar */}
        <div className="w-full h-3 bg-slate-800 rounded-full overflow-hidden p-0.5">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${confidencePct}%` }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
            className={`h-full rounded-full ${
              isSynthetic
                ? 'bg-gradient-to-r from-orange-500 via-red-500 to-rose-600 shadow-md shadow-red-500/50'
                : 'bg-gradient-to-r from-teal-500 via-emerald-500 to-green-500 shadow-md shadow-emerald-500/50'
            }`}
          />
        </div>

        <div className="flex justify-between text-xs text-slate-500 pt-1">
          <span>0% (Uncertain)</span>
          <span>50% (Decision Boundary)</span>
          <span>100% (High Confidence)</span>
        </div>
      </div>

      {/* Visual Explainability & Heatmap Section */}
      <div className="space-y-4">
        {/* Controls Bar */}
        <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-950/40 p-3 rounded-2xl border border-slate-800/60">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5 mr-2">
              <Layers className="w-4 h-4 text-indigo-400" />
              Display Mode:
            </span>
            <div className="flex bg-slate-900 rounded-xl p-1 border border-slate-800">
              <button
                onClick={() => setViewMode('overlay')}
                className={`px-3 py-1 text-xs font-medium rounded-lg transition ${
                  viewMode === 'overlay'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                Heatmap Overlay
              </button>
              <button
                onClick={() => setViewMode('side-by-side')}
                className={`px-3 py-1 text-xs font-medium rounded-lg transition ${
                  viewMode === 'side-by-side'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                Side-by-Side
              </button>
              <button
                onClick={() => setViewMode('raw-heatmap')}
                className={`px-3 py-1 text-xs font-medium rounded-lg transition ${
                  viewMode === 'raw-heatmap'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                Raw Attention
              </button>
            </div>
          </div>

          {/* Jet Heatmap Colormap Indicator */}
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Flame className="w-3.5 h-3.5 text-amber-400" />
            <span>Attention:</span>
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-blue-400">Low</span>
              <div className="w-16 h-2 rounded bg-gradient-to-r from-blue-600 via-yellow-400 to-red-600" />
              <span className="text-[10px] text-red-400">High</span>
            </div>
          </div>
        </div>

        {/* Image Display Area */}
        <div className="relative rounded-2xl overflow-hidden bg-black/80 border border-slate-800 p-3 sm:p-5 flex items-center justify-center min-h-[340px]">
          {viewMode === 'overlay' && (
            <div className="relative max-w-md w-full aspect-square rounded-xl overflow-hidden shadow-2xl border border-slate-700/80">
              <img
                src={originalImageUrl}
                alt="Original Upload"
                className="absolute inset-0 w-full h-full object-cover"
              />
              <img
                src={result.raw_heatmap || result.attention_heatmap}
                alt="ViT attention heatmap overlaid on the original image"
                className="absolute inset-0 w-full h-full object-cover mix-blend-screen"
                style={{ opacity: overlayAlpha }}
              />
              <div className="absolute top-3 left-3 px-2.5 py-1 bg-black/75 backdrop-blur-md rounded-md text-[11px] font-medium text-slate-300 border border-white/10">
                ViT CLS Attention Rollout Overlay
              </div>
              <label className="absolute bottom-3 left-3 right-3 flex items-center gap-3 rounded-lg border border-white/10 bg-black/75 px-3 py-2 text-xs text-slate-200 backdrop-blur">
                <span>Blend</span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={overlayAlpha}
                  onChange={(event) => setOverlayAlpha(Number(event.target.value))}
                  aria-label="Attention overlay opacity"
                  className="w-full accent-indigo-500"
                />
                <span>{Math.round(overlayAlpha * 100)}%</span>
              </label>
            </div>
          )}

          {viewMode === 'side-by-side' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
              <div className="space-y-2">
                <div className="text-xs font-semibold text-slate-400">Original Upload</div>
                <div className="relative aspect-square rounded-xl overflow-hidden border border-slate-800 bg-slate-950">
                  <img
                    src={originalImageUrl}
                    alt="Original Input"
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>
              <div className="space-y-2">
                <div className="text-xs font-semibold text-slate-400">Attention Heatmap</div>
                <div className="relative aspect-square rounded-xl overflow-hidden border border-slate-800 bg-slate-950">
                  <img
                    src={result.attention_heatmap}
                    alt="Attention Map"
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>
            </div>
          )}

          {viewMode === 'raw-heatmap' && (
            <div className="relative max-w-md w-full aspect-square rounded-xl overflow-hidden shadow-2xl border border-slate-700/80">
              <img
                src={result.raw_heatmap || result.attention_heatmap}
                alt="Raw Attention Map"
                className="w-full h-full object-cover"
              />
              <div className="absolute top-3 left-3 px-2.5 py-1 bg-black/75 backdrop-blur-md rounded-md text-[11px] font-medium text-slate-300 border border-white/10">
                Isolated 14×14 Interpolated Self-Attention
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Deepfake & ViT Architecture Insights Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5 pt-2">
        <div className="bg-slate-950/40 border border-slate-800/80 rounded-xl p-4">
          <div className="flex items-center gap-2 text-indigo-400 text-xs font-semibold uppercase mb-1">
            <Grid className="w-3.5 h-3.5" />
            <span>ViT Token Geometry</span>
          </div>
          <div className="text-lg font-bold text-slate-200">
            {result.patch_analysis?.grid_size || '14×14'} ({result.patch_analysis?.patch_count || 196} Patches)
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Each 16×16 px patch linearly embedded into D=768 token vectors.
          </p>
        </div>

        <div className="bg-slate-950/40 border border-slate-800/80 rounded-xl p-4">
          <div className="flex items-center gap-2 text-amber-400 text-xs font-semibold uppercase mb-1">
            <Flame className="w-3.5 h-3.5" />
            <span>Salient Activation Focus</span>
          </div>
          <div className="text-sm font-bold text-slate-200 truncate" title={result.patch_analysis?.highest_activation_zone}>
            {result.patch_analysis?.highest_activation_zone || 'Facial Synthesis Boundary'}
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Top self-attention divergence focused on synthetic pixel transitions.
          </p>
        </div>

        <div className="bg-slate-950/40 border border-slate-800/80 rounded-xl p-4">
          <div className="flex items-center gap-2 text-emerald-400 text-xs font-semibold uppercase mb-1">
            <Info className="w-3.5 h-3.5" />
            <span>Inference Architecture</span>
          </div>
          <div className="text-sm font-bold text-slate-200 truncate">
            {result.model_name || 'vit_base_patch16_224'}
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Mode: <span className="text-indigo-300 font-mono">{result.mode}</span> | Resolution: 224×224 px
          </p>
        </div>
      </div>
    </motion.div>
  );
};
