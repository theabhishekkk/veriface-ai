'use client';

import React from 'react';
import { motion } from 'framer-motion';
import {
  History,
  ShieldAlert,
  ShieldCheck,
  Clock,
  RefreshCw,
  Database,
  ExternalLink,
  FileText
} from 'lucide-react';
import { ScanRecord } from '../lib/supabaseClient';

interface HistorySidebarProps {
  scans: ScanRecord[];
  isLoading: boolean;
  supabaseConnected: boolean;
  onRefresh: () => void;
}

export const HistorySidebar: React.FC<HistorySidebarProps> = ({
  scans,
  isLoading,
  supabaseConnected,
  onRefresh,
}) => {
  const formatTimestamp = (isoStr?: string) => {
    if (!isoStr) return 'Just now';
    try {
      const date = new Date(isoStr);
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) +
        ' · ' +
        date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    } catch {
      return 'Recent';
    }
  };

  return (
    <div className="w-full bg-slate-900/60 border border-slate-800 rounded-3xl p-6 backdrop-blur-xl shadow-xl space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-indigo-950/80 border border-indigo-500/30 text-indigo-400">
            <History className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-slate-100 flex items-center gap-2">
              Recent Scans
            </h3>
            <p className="text-xs text-slate-400 flex items-center gap-1.5">
              <Database className="w-3 h-3 text-indigo-400" />
              <span>
                {supabaseConnected ? 'Supabase PostgreSQL' : 'Recent scan history'}
              </span>
            </p>
          </div>
        </div>

        <button
          onClick={onRefresh}
          disabled={isLoading}
          className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white border border-slate-700/60 transition disabled:opacity-50"
          title="Refresh History"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-indigo-400' : ''}`} />
        </button>
      </div>

      {/* History Items List */}
      <div className="space-y-3">
        {scans.length === 0 ? (
          <div className="py-8 text-center text-slate-500 text-xs flex flex-col items-center justify-center">
            <Clock className="w-8 h-8 text-slate-600 mb-2 stroke-[1.5]" />
            <p className="font-medium">No previous scans found</p>
            <p className="text-slate-600 mt-0.5">Upload a photo to see results recorded in real-time.</p>
          </div>
        ) : (
          scans.slice(0, 5).map((scan, index) => {
            const isSynthetic = scan.is_deepfake || scan.label.toLowerCase() === 'synthetic';
            const confidencePercent = Math.round((scan.confidence || 0) * 100);

            return (
              <motion.div
                key={scan.id || `scan-${index}`}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                className="group relative bg-slate-950/50 hover:bg-slate-950/80 border border-slate-800/80 hover:border-slate-700 rounded-2xl p-3.5 transition-all flex items-center justify-between gap-3"
              >
                <div className="flex items-center gap-3 min-w-0">
                  <div
                    className={`p-2 rounded-xl flex-shrink-0 ${
                      isSynthetic
                        ? 'bg-red-950/70 border border-red-500/40 text-red-400'
                        : 'bg-emerald-950/70 border border-emerald-500/40 text-emerald-400'
                    }`}
                  >
                    {isSynthetic ? (
                      <ShieldAlert className="w-4 h-4" />
                    ) : (
                      <ShieldCheck className="w-4 h-4" />
                    )}
                  </div>

                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span
                        className={`text-xs font-semibold px-2 py-0.5 rounded-md ${
                          isSynthetic
                            ? 'bg-red-900/40 text-red-300 border border-red-800/50'
                            : 'bg-emerald-900/40 text-emerald-300 border border-emerald-800/50'
                        }`}
                      >
                        {isSynthetic ? 'Synthetic' : 'Authentic'}
                      </span>
                      <span className="text-xs font-bold text-slate-200">
                        {confidencePercent}%
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-400 truncate max-w-[170px] mt-1 flex items-center gap-1 font-mono">
                      <FileText className="w-2.5 h-2.5 text-slate-500" />
                      <span className="truncate">{scan.filename}</span>
                    </div>
                  </div>
                </div>

                <div className="text-right flex-shrink-0">
                  <div className="text-[10px] text-slate-400">
                    {formatTimestamp(scan.created_at)}
                  </div>
                  {scan.execution_time_ms ? (
                    <div className="text-[10px] text-slate-500 font-mono mt-0.5">
                      {Math.round(scan.execution_time_ms)}ms
                    </div>
                  ) : null}
                </div>
              </motion.div>
            );
          })
        )}
      </div>

      {/* Supabase status footer indicator */}
      <div className="pt-2 text-[11px] text-slate-500 flex items-center justify-between border-t border-slate-800/60">
        <span className="flex items-center gap-1.5">
          <span
            className={`w-2 h-2 rounded-full ${
              supabaseConnected ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'
            }`}
          />
          {supabaseConnected ? 'Connected to PostgreSQL' : 'In-memory fallback'}
        </span>
        <span className="font-mono text-slate-500">v1.0.0</span>
      </div>
    </div>
  );
};
