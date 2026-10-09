import { createClient, SupabaseClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || '';
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '';

export interface ScanRecord {
  id?: string;
  filename: string;
  label: 'Real' | 'Synthetic';
  is_deepfake: boolean;
  confidence: number;
  execution_time_ms: number;
  created_at?: string;
}

// Initialize client only if valid configuration exists; otherwise provide null-safe stub
export const supabase: SupabaseClient | null = (supabaseUrl && supabaseAnonKey)
  ? createClient(supabaseUrl, supabaseAnonKey)
  : null;

export const isSupabaseConfigured = (): boolean => {
  return Boolean(supabaseUrl && supabaseAnonKey);
};

/**
 * Fetch 5 most recent scans from Supabase or fallback backend endpoint
 */
export async function fetchRecentScans(limit: number = 5): Promise<ScanRecord[]> {
  const backendApiUrl = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000').replace(/\/$/, '');

  try {
    const response = await fetch(`${backendApiUrl}/scans?limit=${limit}`, {
      cache: 'no-store',
    });
    if (response.ok) {
      return (await response.json()) as ScanRecord[];
    }
    console.warn(`[VeriFace] Backend history request failed (${response.status}).`);
  } catch (error) {
    console.warn('[VeriFace] Backend history request failed:', error);
  }

  if (supabase) {
    const { data, error } = await supabase
      .from('scans')
      .select('id, filename, label, is_deepfake, confidence, execution_time_ms, created_at')
      .order('created_at', { ascending: false })
      .limit(limit);
    if (error) {
      console.warn('[VeriFace] Supabase history request failed:', error.message);
    } else if (data) {
      return data as ScanRecord[];
    }
  }

  return [];
}

/**
 * Insert scan record into Supabase if client is configured
 */
export async function logScanToSupabase(record: ScanRecord): Promise<boolean> {
  if (!supabase) return false;

  try {
    const { error } = await supabase.from('scans').insert([
      {
        filename: record.filename,
        label: record.label,
        is_deepfake: record.is_deepfake,
        confidence: record.confidence,
        execution_time_ms: record.execution_time_ms,
        created_at: record.created_at || new Date().toISOString(),
      },
    ]);

    return !error;
  } catch (err) {
    console.error('[VeriFace] Failed to record scan to Supabase:', err);
    return false;
  }
}
