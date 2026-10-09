import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'VeriFace AI | Vision Transformer Deepfake Detection',
  description:
    'State-of-the-art deepfake detection and media integrity verification powered by Vision Transformers (ViT) and attention rollout explainability.',
  keywords: ['deepfake detection', 'vision transformer', 'vit', 'attention map', 'ai forensics', 'facial integrity'],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#080c16] text-slate-100 antialiased selection:bg-indigo-500 selection:text-white">
        {children}
      </body>
    </html>
  );
}
