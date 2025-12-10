import React, { useState } from 'react';
import ImageUpload from './components/ImageUpload';
import ResultsOverlay from './components/ResultsOverlay';
import './App.css';

// Define types matching new backend response
interface PestBox {
    box: number[]; // [x, y, w, h]
    label: string;
    confidence: number;
    class_id: number;
}

interface AnalysisResult {
    pest: {
        count: number;
        severity: string;
        boxes: PestBox[];
    };
    disease: {
        diseased_area_percent: number;
        severity: string;
        mask_b64: string | null;
        mask_shape: number[];
    };
}


function App() {
    const [selectedImage, setSelectedImage] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [result, setResult] = useState<AnalysisResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleImageSelect = (file: File) => {
        setSelectedImage(file);
        setPreviewUrl(URL.createObjectURL(file));
        setResult(null);
        setError(null);
    };

    const handleAnalyze = async () => {
        if (!selectedImage) return;

        setLoading(true);
        setError(null);

        const formData = new FormData();
        formData.append('file', selectedImage);

        try {
            const response = await fetch('http://localhost:8000/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Analysis failed');
            }

            const data = await response.json();
            setResult(data);
        } catch (err) {
            console.error(err);
            setError('Failed to process image. Ensure backend is running.');
        } finally {
            setLoading(false);
        }
    };

    const getSeverityClass = (severity: string) => {
        const s = severity.toLowerCase();
        if (s.includes('high') || s.includes('critical')) return 'badge danger';
        if (s.includes('moderate')) return 'badge warning';
        return 'badge success';
    };

    return (
        <div className="container">
            <header className="header">
                <h1>Agricultural Visual Intelligence</h1>
                <p>Multi-task Pest and Disease Analysis</p>
            </header>

            <div className="main-grid">
                {/* Left Column: Input and Controls */}
                <div className="left-column">
                    <div className="card">
                        <h2 className="card-title">1. Upload Plant Image</h2>
                        <ImageUpload onImageSelect={handleImageSelect} isLoading={loading} />

                        {selectedImage && (
                            <div className="btn-container">
                                <button
                                    onClick={handleAnalyze}
                                    disabled={loading}
                                    className="btn"
                                >
                                    {loading ? 'Processing...' : 'Run Analysis'}
                                </button>
                            </div>
                        )}
                        {error && (
                            <div className="error-msg">
                                {error}
                            </div>
                        )}
                    </div>

                    {/* Stats Cards (Visible after result) */}
                    {result && (
                        <div className="results-grid">
                            <div className="card">
                                <h3 className="sub-title">Pest Detection</h3>
                                <div style={{ display: 'flex', alignItems: 'baseline' }}>
                                    <span className="stat-value">{result.pest.count}</span>
                                    <span className="stat-label">pests detected</span>
                                </div>
                                <div className={getSeverityClass(result.pest.severity)}>
                                    Severity: {result.pest.severity}
                                </div>
                            </div>

                            <div className="card">
                                <h3 className="sub-title">Disease Analysis</h3>
                                <div style={{ display: 'flex', alignItems: 'baseline' }}>
                                    <span className="stat-value">{result.disease.diseased_area_percent}%</span>
                                    <span className="stat-label">infected area</span>
                                </div>
                                <div className={getSeverityClass(result.disease.severity)}>
                                    Severity: {result.disease.severity}
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Right Column: Visualization */}
                <div className="card viz-container">
                    <h2 className="card-title">2. Visualization</h2>
                    <ResultsOverlay
                        imageUrl={previewUrl}
                        boxes={result?.pest.boxes || []}
                        maskBase64={result?.disease.mask_b64 || null}
                        maskShape={result?.disease.mask_shape || null}
                    />
                    <p className="viz-note">
                        (Overlays: Red Boxes = Pests, White/Tinted Regions = Disease Mask)
                    </p>
                </div>
            </div>
        </div>
    );
}

export default App;
