'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  Shield,
  Cpu,
  Layers,
  Sparkles,
  Activity,
  AlertCircle,
  Zap,
  Info
} from 'lucide-react';

import { Uploader } from '../components/Uploader';
import { ResultsDashboard, PredictResult } from '../components/ResultsDashboard';
import { HistorySidebar } from '../components/HistorySidebar';
import { fetchRecentScans, ScanRecord } from '../lib/supabaseClient';

export default function Home() {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [predictResult, setPredictResult] = useState<PredictResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Scans history state
  const [scansHistory, setScansHistory] = useState<ScanRecord[]>([]);
  const [isHistoryLoading, setIsHistoryLoading] = useState<boolean>(false);

  // Backend status
  const [backendHealth, setBackendHealth] = useState<{
    online: boolean;
    mode: string;
    supabase: boolean;
  }>({ online: false, mode: 'unknown', supabase: false });

  const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  // Check backend health on mount
  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${backendUrl}/health`, { method: 'GET' });
      if (res.ok) {
        const data = await res.json();
        setBackendHealth({
          online: true,
          mode: data.model_mode || 'vit_base_patch16_224',
          supabase: Boolean(data.supabase_connected),
        });
        return;
      }
    } catch {
      // Backend not yet reachable
    }
    setBackendHealth({ online: false, mode: 'offline', supabase: false });
  }, [backendUrl]);

  // Load scan history
  const loadHistory = useCallback(async () => {
    setIsHistoryLoading(true);
    try {
      const data = await fetchRecentScans(5);
      setScansHistory(data);
    } catch (err) {
      console.warn('Failed to load history:', err);
    } finally {
      setIsHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    loadHistory();
  }, [checkHealth, loadHistory]);

  // Handle image upload and trigger ViT prediction
  const handleImageSelected = async (file: File, preview: string) => {
    setPreviewUrl(preview);
    setPredictResult(null);
    setError(null);
    setIsLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${backendUrl}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Inference server returned error (${res.status})`);
      }

      const data: PredictResult = await res.json();
      setPredictResult(data);

      // Refresh recent scans from database
      setTimeout(() => {
        loadHistory();
      }, 500);
    } catch (err: unknown) {
      console.error('Prediction failed:', err);
      setError(
        (err instanceof Error && err.message) ||
          'Failed to connect to backend inference engine. Ensure the FastAPI backend is running.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setPreviewUrl(null);
    setPredictResult(null);
    setError(null);
  };

  return (
    <main className="min-h-screen text-slate-100 flex flex-col justify-between">
      {/* Top Navigation */}
      <header className="border-b border-slate-800/80 bg-slate-950/60 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/25">
              <Shield className="w-5 h-5 text-white" />
            </div>
            <div>
              <span className="font-extrabold text-lg tracking-tight bg-gradient-to-r from-white via-slate-100 to-indigo-300 bg-clip-text text-transparent">
                VeriFace AI
              </span>
              <span className="ml-2 px-2 py-0.5 rounded-full text-[10px] font-semibold tracking-wider uppercase bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                ViT Engine
              </span>
            </div>
          </div>

          {/* Engine & Database Status Indicators */}
          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900/80 border border-slate-800 text-xs">
              <span
                className={`w-2 h-2 rounded-full ${
                  backendHealth.online
                    ? backendHealth.mode === 'mock'
                      ? 'bg-amber-400 animate-pulse'
                      : 'bg-emerald-400 animate-pulse'
                    : 'bg-rose-500'
                }`}
              />
              <span className="text-slate-400">
                {backendHealth.online
                  ? backendHealth.mode === 'mock'
                    ? 'Backend: Mock Engine'
                    : 'Backend: PyTorch ViT-Base'
                  : 'Backend: Connecting...'}
              </span>
            </div>

            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-indigo-950/60 border border-indigo-500/30 text-indigo-300 text-xs">
              <Cpu className="w-3.5 h-3.5 text-indigo-400" />
              <span className="font-mono">16×16 Patches</span>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 flex-1 w-full space-y-10">
        <div className="text-center max-w-3xl mx-auto space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-950/50 border border-indigo-500/30 text-indigo-300 text-xs font-medium">
            <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
            <span>Vision Transformer Media Forensics & Explainability</span>
          </div>
          <h1 className="text-3xl sm:text-5xl font-black tracking-tight text-white">
            AI-Based Deepfake Detection using{' '}
            <span className="bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
              Vision Transformers
            </span>
          </h1>
          <p className="text-slate-400 text-sm sm:text-base leading-relaxed">
            Upload any face portrait to decompose image patches into multi-head self-attention
            vectors, detect generative artifacts, and inspect visual attention explainability heatmaps.
          </p>
        </div>

        {backendHealth.mode === 'mock' && (
          <div className="max-w-4xl mx-auto rounded-xl border border-amber-500/30 bg-amber-950/30 px-4 py-3 text-center text-sm text-amber-200">
            Mock mode is enabled: results are simulated and are not suitable for forensic or production decisions.
          </div>
        )}

        {/* Global Error Banner */}
        {error && (
          <div className="max-w-4xl mx-auto p-4 bg-red-950/70 border border-red-800 rounded-2xl flex items-start gap-3 text-red-200">
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
            <div className="text-sm">
              <p className="font-semibold text-red-100">Inference Request Error</p>
              <p className="text-red-300/90 mt-0.5">{error}</p>
              <p className="text-xs text-red-400 mt-2 font-mono">
                Tip: Run `python main.py` in /backend or `docker-compose up` to start the backend server.
              </p>
            </div>
          </div>
        )}

        {/* Core Interactive Zone */}
        <div className="max-w-4xl mx-auto">
          {predictResult && previewUrl ? (
            <ResultsDashboard
              originalImageUrl={previewUrl}
              result={predictResult}
              onReset={handleReset}
            />
          ) : (
            <Uploader
              onImageSelected={handleImageSelected}
              isLoading={isLoading}
            />
          )}
        </div>

        {/* Feature Capabilities Grid (Shown when not viewing full result) */}
        {!predictResult && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5 max-w-5xl mx-auto pt-4">
            <div className="p-5 rounded-2xl bg-slate-900/40 border border-slate-800 hover:border-indigo-500/40 transition">
              <div className="w-10 h-10 rounded-xl bg-indigo-950/80 border border-indigo-500/30 flex items-center justify-center text-indigo-400 mb-3">
                <Cpu className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-slate-200 mb-1">
                vit_base_patch16_224
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Splits images into a 14×14 grid of 16×16 non-overlapping patches, computing self-attention
                across 12 transformer encoder blocks.
              </p>
            </div>

            <div className="p-5 rounded-2xl bg-slate-900/40 border border-slate-800 hover:border-purple-500/40 transition">
              <div className="w-10 h-10 rounded-xl bg-purple-950/80 border border-purple-500/30 flex items-center justify-center text-purple-400 mb-3">
                <Layers className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-slate-200 mb-1">
                Attention Rollout Heatmap
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Hooks into the final multi-head attention weights to compute token saliency and
                overlays a high-resolution JET colormap on facial seams.
              </p>
            </div>

            <div className="p-5 rounded-2xl bg-slate-900/40 border border-slate-800 hover:border-emerald-500/40 transition">
              <div className="w-10 h-10 rounded-xl bg-emerald-950/80 border border-emerald-500/30 flex items-center justify-center text-emerald-400 mb-3">
                <Activity className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-slate-200 mb-1">
                Supabase Scan Auditing
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Every classification, confidence score, and inference latency is logged to PostgreSQL
                for real-time forensic history tracking.
              </p>
            </div>
          </div>
        )}

        {/* Scan History Section (5 Most Recent Scans) */}
        <div className="max-w-4xl mx-auto pt-6">
          <HistorySidebar
            scans={scansHistory}
            isLoading={isHistoryLoading}
            supabaseConnected={backendHealth.supabase}
            onRefresh={loadHistory}
          />
        </div>
      </div>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950/80 py-6 mt-12 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-400">VeriFace AI Platform</span>
            <span>·</span>
            <span>Vision Transformers for Deepfake Detection</span>
          </div>
          <div className="flex items-center gap-4 text-slate-400">
            <span>FastAPI + timm + Next.js + Supabase</span>
          </div>
        </div>
      </footer>
    </main>
  );
}
