'use client';

import React, { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  UploadCloud,
  Camera,
  Image as ImageIcon,
  AlertCircle,
  RefreshCw,
  Cpu,
  Eye,
  Check,
  X
} from 'lucide-react';

interface UploaderProps {
  onImageSelected: (file: File, previewUrl: string) => void;
  isLoading: boolean;
}

export const Uploader: React.FC<UploaderProps> = ({ onImageSelected, isLoading }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [showWebcam, setShowWebcam] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);

  React.useEffect(() => {
    return () => {
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  // Cycle through inference loading messages
  React.useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isLoading) {
      setLoadingStep(0);
      interval = setInterval(() => {
        setLoadingStep((prev) => (prev + 1) % 3);
      }, 900);
    }
    return () => clearInterval(interval);
  }, [isLoading]);

  const steps = [
    'Rescaling image to 224×224 ViT patch matrix...',
    'Evaluating 196 attention heads across 12 transformer layers...',
    'Synthesizing multi-focal explainability attention rollout...',
  ];

  // Process file validation & selection
  const processFile = useCallback(
    (file: File) => {
      setErrorMsg(null);
      const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
      if (!validTypes.includes(file.type)) {
        setErrorMsg('Please upload a valid JPEG or PNG image (.jpg or .png).');
        return;
      }
      if (file.size > 10 * 1024 * 1024) {
        setErrorMsg('Image size exceeds the 10MB limit. Please upload a smaller file.');
        return;
      }

      const previewUrl = URL.createObjectURL(file);
      onImageSelected(file, previewUrl);
    },
    [onImageSelected]
  );

  // Drag and drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
    e.target.value = '';
  };

  // Webcam controls
  const startWebcam = async () => {
    setErrorMsg(null);
    setShowWebcam(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 640 } },
      });
      mediaStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setCameraActive(true);
      }
    } catch {
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
      setErrorMsg('Unable to access webcam. Please check camera permissions or upload an image.');
      setShowWebcam(false);
      setCameraActive(false);
    }
  };

  const stopWebcam = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    setShowWebcam(false);
    setCameraActive(false);
  };

  const captureFrame = () => {
    if (!videoRef.current) return;

    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 640;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const file = new File([blob], `webcam_capture_${Date.now()}.jpg`, {
          type: 'image/jpeg',
        });
        stopWebcam();
        processFile(file);
      },
      'image/jpeg',
      0.92
    );
  };

  return (
    <div className="w-full">
      {/* Error alert */}
      <AnimatePresence>
        {errorMsg && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="mb-4 p-3 bg-red-950/60 border border-red-700/50 rounded-xl flex items-center justify-between text-red-200 text-sm"
          >
            <div className="flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
              <span>{errorMsg}</span>
            </div>
            <button
              onClick={() => setErrorMsg(null)}
              className="text-red-400 hover:text-white p-1 rounded-md"
            >
              <X className="w-4 h-4" />
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Webcam modal / inline panel */}
      {showWebcam ? (
        <div className="relative bg-slate-900/90 border border-indigo-500/40 rounded-2xl p-6 overflow-hidden shadow-2xl backdrop-blur-md">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2 text-indigo-300 font-medium text-sm">
              <Camera className="w-4 h-4 text-indigo-400 animate-pulse" />
              <span>Live Facial Capture</span>
            </div>
            <button
              onClick={stopWebcam}
              className="p-1.5 rounded-lg bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700 transition"
              title="Close Webcam"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="relative aspect-square max-w-sm mx-auto rounded-xl overflow-hidden bg-black border border-slate-700/80">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover mirror-mode"
            />
            {/* Alignment crosshair */}
            <div className="absolute inset-0 border-2 border-indigo-500/25 pointer-events-none rounded-xl m-8 flex items-center justify-center">
              <div className="w-24 h-24 border border-dashed border-indigo-400/40 rounded-full animate-pulse-slow" />
            </div>
          </div>

          <div className="mt-5 flex items-center justify-center gap-3">
            <button
              onClick={captureFrame}
              disabled={!cameraActive}
              className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium rounded-xl flex items-center gap-2 shadow-lg shadow-indigo-600/30 transition transform active:scale-95"
            >
              <Camera className="w-4 h-4" />
              <span>Capture & Analyze</span>
            </button>
            <button
              onClick={stopWebcam}
              className="px-4 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm font-medium rounded-xl transition"
            >
              Cancel
            </button>
          </div>
        </div>
      ) : (
        /* Main Drag & Drop Zone */
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !isLoading && fileInputRef.current?.click()}
          className={`relative group cursor-pointer border-2 border-dashed rounded-3xl p-8 sm:p-12 transition-all duration-300 backdrop-blur-md overflow-hidden ${
            isDragOver
              ? 'border-indigo-400 bg-indigo-950/30 glow-primary scale-[1.01]'
              : 'border-slate-700/80 hover:border-indigo-500/60 bg-slate-900/40 hover:bg-slate-900/60'
          }`}
        >
          {/* Subtle animated background gradient */}
          <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500/10 via-purple-500/10 to-emerald-500/10 rounded-3xl blur-xl opacity-50 group-hover:opacity-100 transition duration-500 pointer-events-none" />

          <input
            ref={fileInputRef}
            type="file"
            accept=".jpg,.jpeg,.png"
            onChange={handleFileInputChange}
            className="hidden"
            disabled={isLoading}
          />

          {isLoading ? (
            /* Loading State */
            <div className="flex flex-col items-center justify-center py-6 text-center">
              <div className="relative mb-6">
                <div className="w-20 h-20 rounded-2xl bg-indigo-950/80 border border-indigo-500/50 flex items-center justify-center glow-primary animate-pulse">
                  <Cpu className="w-10 h-10 text-indigo-400 animate-spin" />
                </div>
                <div className="absolute -bottom-1 -right-1 p-1 bg-emerald-600 rounded-full">
                  <Eye className="w-3.5 h-3.5 text-white" />
                </div>
              </div>

              <h4 className="text-lg font-semibold text-white mb-2">
                Vision Transformer Processing
              </h4>
              <motion.p
                key={loadingStep}
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="text-sm text-indigo-300/90 max-w-md h-6"
              >
                {steps[loadingStep]}
              </motion.p>

              {/* Progress bar pulse */}
              <div className="w-64 h-1.5 bg-slate-800 rounded-full mt-6 overflow-hidden">
                <div className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full animate-pulse-slow w-full" />
              </div>
            </div>
          ) : (
            /* Normal Empty State */
            <div className="flex flex-col items-center justify-center text-center relative z-10">
              <div className="w-16 h-16 sm:w-20 sm:h-20 mb-5 rounded-2xl bg-gradient-to-tr from-indigo-950 to-slate-800 border border-indigo-500/30 flex items-center justify-center group-hover:scale-110 group-hover:border-indigo-400/60 transition-transform duration-300 shadow-lg shadow-indigo-950/50">
                <UploadCloud className="w-8 h-8 sm:w-10 sm:h-10 text-indigo-400 group-hover:text-indigo-300 transition-colors" />
              </div>

              <h3 className="text-lg sm:text-xl font-semibold text-slate-100 mb-2">
                Drag & drop facial image here
              </h3>
              <p className="text-sm text-slate-400 max-w-sm mb-6">
                Supports JPG and PNG. Processed at 224×224 with ViT-Base 16×16 patch attention.
              </p>

              <div className="flex flex-wrap items-center justify-center gap-3">
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    fileInputRef.current?.click();
                  }}
                  className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-xl shadow-lg shadow-indigo-600/25 transition-all flex items-center gap-2 transform active:scale-95"
                >
                  <ImageIcon className="w-4 h-4" />
                  <span>Browse Device</span>
                </button>

                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    startWebcam();
                  }}
                  className="px-5 py-2.5 bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700 text-slate-200 text-sm font-medium rounded-xl transition-all flex items-center gap-2 hover:border-indigo-500/50 transform active:scale-95"
                >
                  <Camera className="w-4 h-4 text-indigo-400" />
                  <span>Use Camera</span>
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
