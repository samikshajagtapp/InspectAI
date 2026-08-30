import React, { useState, useEffect, useRef } from 'react';
import {
  Upload,
  ShieldCheck,
  AlertOctagon,
  CheckCircle2,
  XCircle,
  Eye,
  RefreshCw,
  Sliders,
  Maximize2,
  X,
  FileText,
  Layers,
  Activity,
  Cpu,
  Target,
  FileBarChart,
  Box,
  CheckSquare,
  AlertTriangle
} from 'lucide-react';

const API_BASE = 'http://127.0.0.1:5001/api';

export default function App() {
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [threshold, setThreshold] = useState(10.46);
  const [activeTab, setActiveTab] = useState('red_marked'); // red_marked | heatmap | overlay | visualization
  const [modalImage, setModalImage] = useState(null);
  const [health, setHealth] = useState({ loaded: false, backbone: 'resnet18', accelerator: 'mps' });

  const fileInputRef = useRef(null);

  useEffect(() => {
    fetchHealth();
    fetchSamples();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        setHealth({
          loaded: data.model_loaded,
          backbone: data.backbone || 'resnet18',
          accelerator: data.accelerator || 'mps',
        });
      }
    } catch (e) {
      console.warn('Backend server connection error:', e);
    }
  };

  const fetchSamples = async () => {
    try {
      const res = await fetch(`${API_BASE}/samples`);
      if (res.ok) {
        const data = await res.json();
        setSamples(data.samples || []);
      }
    } catch (e) {
      console.warn('Error fetching sample list:', e);
    }
  };

  const handleFileUpload = (file) => {
    if (!file) return;
    setSelectedSample(null);
    runInference({ file });
  };

  const handleSampleClick = (sample) => {
    setSelectedSample(sample);
    runInference({ sample_path: sample.path });
  };

  const runInference = async ({ file, sample_path }) => {
    setLoading(true);
    setError(null);
    setReport(null);

    const formData = new FormData();
    if (threshold !== null) {
      formData.append('threshold', threshold.toString());
    }

    if (file) {
      formData.append('file', file);
    } else if (sample_path) {
      formData.append('sample_path', sample_path);
    }

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      if (res.ok && data.status === 'success') {
        setReport(data);
        setActiveTab(data.is_defective ? 'red_marked' : 'visualization');
      } else {
        setError(data.message || 'Model execution failed.');
      }
    } catch (e) {
      setError('Connection failure: Unable to communicate with SAP BTP AI service backend.');
    } finally {
      setLoading(false);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const reRunThreshold = (newThresh) => {
    setThreshold(newThresh);
    if (selectedSample) {
      runInference({ sample_path: selectedSample.path });
    }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--sap-bg)' }}>
      
      {/* 1. SAP Shell Bar Header */}
      <header className="sap-shell-header">
        <div
          style={{
            maxWidth: '1360px',
            margin: '0 auto',
            padding: '0.85rem 1.5rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          {/* SAP Brand Branding */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div
              style={{
                background: '#0070F2',
                color: '#ffffff',
                fontWeight: 900,
                fontSize: '0.95rem',
                padding: '0.3rem 0.6rem',
                borderRadius: '4px',
                letterSpacing: '0.05em',
              }}
            >
              SAP
            </div>
            <div>
              <h1 style={{ fontSize: '1.15rem', fontWeight: 700, lineHeight: 1.2 }}>
                Business Technology Platform <span style={{ fontWeight: 400, color: '#a0b3c6' }}>| Quality Inspection</span>
              </h1>
              <p style={{ fontSize: '0.75rem', color: '#b0c4d8' }}>
                Automated Pharmaceutical Bottle Anomaly Detection System
              </p>
            </div>
          </div>

          {/* System Runtime Metadata Badges */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div
              style={{
                background: 'rgba(255, 255, 255, 0.08)',
                border: '1px solid rgba(255, 255, 255, 0.15)',
                borderRadius: '4px',
                padding: '0.35rem 0.75rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                fontSize: '0.78rem',
                color: '#e1e9f0',
              }}
            >
              <Cpu size={14} color="#0070F2" />
              <span>Model: <strong>PatchCore ({health.backbone})</strong></span>
              <span style={{ color: 'rgba(255, 255, 255, 0.2)' }}>|</span>
              <Activity size={14} color="#107C41" />
              <span>Acceleration: <strong>{health.accelerator.toUpperCase()}</strong></span>
            </div>
          </div>
        </div>
      </header>

      {/* Main SAP Fiori Page Layout */}
      <main style={{ maxWidth: '1360px', width: '100%', margin: '0 auto', padding: '2rem 1.5rem', flex: 1 }}>
        
        {/* Section Header */}
        <div style={{ marginBottom: '1.75rem', borderBottom: '1px solid var(--sap-card-border)', paddingBottom: '1rem' }}>
          <h2 style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--sap-text-primary)', marginBottom: '0.25rem' }}>
            Inspection Input & Sample Dataset Selection
          </h2>
          <p style={{ fontSize: '0.88rem', color: 'var(--sap-text-secondary)' }}>
            Upload a pharmaceutical container image from the directory or select a pre-loaded quality test sample.
          </p>
        </div>

        {/* 2. Upload & Sample Gallery Cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
          
          {/* SAP Enterprise File Dropzone */}
          <div
            className={`sap-card ${isDragging ? 'dragging' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            style={{
              padding: '2rem',
              border: isDragging ? '2px dashed var(--sap-brand-blue)' : '2px dashed var(--sap-card-border)',
              background: isDragging ? 'rgba(0, 112, 242, 0.04)' : '#ffffff',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              minHeight: '240px',
              textAlign: 'center',
              position: 'relative',
            }}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              accept="image/*"
              style={{ display: 'none' }}
              onChange={(e) => e.target.files && handleFileUpload(e.target.files[0])}
            />
            <div
              style={{
                width: '56px',
                height: '56px',
                borderRadius: '50%',
                background: 'rgba(0, 112, 242, 0.08)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '1rem',
                color: 'var(--sap-brand-blue)',
              }}
            >
              <Upload size={26} />
            </div>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--sap-text-primary)', marginBottom: '0.3rem' }}>
              Drag & Drop Inspection File
            </h3>
            <p style={{ fontSize: '0.82rem', color: 'var(--sap-text-muted)', marginBottom: '1.2rem' }}>
              Supports standard image formats (PNG, JPG, JPEG, WEBP)
            </p>
            <button
              style={{
                background: 'var(--sap-brand-blue)',
                color: '#ffffff',
                fontWeight: 600,
                border: 'none',
                padding: '0.55rem 1.3rem',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '0.88rem',
              }}
            >
              Select File from Directory
            </button>
          </div>

          {/* Sample Presets Gallery */}
          <div className="sap-card" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--sap-text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Box size={18} color="var(--sap-brand-blue)" /> Inspection Test Samples
              </h3>
              <span style={{ fontSize: '0.75rem', color: 'var(--sap-text-muted)' }}>Pre-configured Test Objects</span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', flex: 1 }}>
              {samples.map((sample) => {
                const isSelected = selectedSample?.id === sample.id;
                return (
                  <div
                    key={sample.id}
                    onClick={() => handleSampleClick(sample)}
                    style={{
                      border: isSelected ? '2px solid var(--sap-brand-blue)' : '1px solid var(--sap-card-border)',
                      background: isSelected ? 'rgba(0, 112, 242, 0.05)' : '#ffffff',
                      borderRadius: '6px',
                      padding: '0.6rem',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.6rem',
                    }}
                    className="sap-card-hover"
                  >
                    {sample.image_b64 ? (
                      <img
                        src={sample.image_b64}
                        alt={sample.title}
                        style={{ width: '44px', height: '44px', objectFit: 'cover', borderRadius: '4px' }}
                      />
                    ) : (
                      <div style={{ width: '44px', height: '44px', background: '#f0f4f8', borderRadius: '4px' }} />
                    )}
                    <div>
                      <h4 style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--sap-text-primary)' }}>{sample.title}</h4>
                      <span
                        style={{
                          fontSize: '0.72rem',
                          color: sample.category === 'good' ? 'var(--sap-semantic-positive)' : 'var(--sap-semantic-negative)',
                          fontWeight: 600,
                        }}
                      >
                        {sample.category === 'good' ? 'Conforming' : 'Defect Sample'}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="sap-card" style={{ padding: '3rem', textAlign: 'center', marginBottom: '2rem' }}>
            <div className="sap-spinner" style={{ margin: '0 auto 1rem auto' }} />
            <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: 'var(--sap-text-primary)', marginBottom: '0.4rem' }}>
              Executing PatchCore Feature Extraction & Vector Evaluation...
            </h3>
            <p style={{ color: 'var(--sap-text-secondary)', fontSize: '0.88rem' }}>
              Calculating feature distances against registered normal memory bank
            </p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div
            className="sap-status-negative"
            style={{
              borderRadius: '6px',
              padding: '1rem 1.25rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.75rem',
              marginBottom: '2rem',
              fontSize: '0.9rem',
            }}
          >
            <AlertOctagon size={20} />
            <span>{error}</span>
          </div>
        )}

        {/* 3. SAP Fiori Inspection Report Object Page */}
        {report && !loading && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
            
            {/* SAP Object Header Summary Tile */}
            <div
              className={`sap-card ${report.is_defective ? 'sap-status-negative' : 'sap-status-positive'}`}
              style={{
                borderRadius: '8px',
                padding: '1.5rem 1.75rem',
              }}
            >
              <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '1.5rem' }}>
                
                {/* Status Indicator */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <div
                    style={{
                      width: '52px',
                      height: '52px',
                      borderRadius: '8px',
                      background: report.is_defective ? '#ffffff' : '#ffffff',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: report.is_defective ? 'var(--sap-semantic-negative)' : 'var(--sap-semantic-positive)',
                      border: report.is_defective ? '1px solid var(--sap-semantic-negative-border)' : '1px solid var(--sap-semantic-positive-border)',
                    }}
                  >
                    {report.is_defective ? <XCircle size={32} /> : <CheckCircle2 size={32} />}
                  </div>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.15rem' }}>
                      <span
                        style={{
                          fontSize: '0.75rem',
                          fontWeight: 700,
                          padding: '0.15rem 0.5rem',
                          borderRadius: '3px',
                          background: report.is_defective ? 'var(--sap-semantic-negative)' : 'var(--sap-semantic-positive)',
                          color: '#ffffff',
                          letterSpacing: '0.04em',
                        }}
                      >
                        {report.is_defective ? 'DEFECTIVE' : 'NORMAL'}
                      </span>
                      <span style={{ fontSize: '0.8rem', color: 'var(--sap-text-secondary)' }}>
                        Quality Verdict
                      </span>
                    </div>
                    <h3 style={{ fontSize: '1.5rem', fontWeight: 800 }}>
                      {report.is_defective ? 'NON-CONFORMING BOTTLE DETECTED' : 'CONFORMING QUALITY INSPECTION'}
                    </h3>
                  </div>
                </div>

                {/* Metric Tiles */}
                <div style={{ display: 'flex', gap: '2.5rem', flexWrap: 'wrap' }}>
                  <div>
                    <span style={{ fontSize: '0.78rem', color: 'var(--sap-text-secondary)', display: 'block', marginBottom: '0.15rem' }}>
                      Anomaly Score
                    </span>
                    <span
                      style={{
                        fontSize: '1.6rem',
                        fontWeight: 800,
                        color: report.is_defective ? 'var(--sap-semantic-negative)' : 'var(--sap-semantic-positive)',
                      }}
                    >
                      {report.anomaly_score}
                    </span>
                  </div>

                  <div>
                    <span style={{ fontSize: '0.78rem', color: 'var(--sap-text-secondary)', display: 'block', marginBottom: '0.15rem' }}>
                      Configured Threshold
                    </span>
                    <span style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--sap-text-primary)' }}>
                      {report.threshold}
                    </span>
                  </div>

                  <div>
                    <span style={{ fontSize: '0.78rem', color: 'var(--sap-text-secondary)', display: 'block', marginBottom: '0.15rem' }}>
                      Processing Latency
                    </span>
                    <span style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--sap-brand-blue)' }}>
                      {report.latency_ms} <span style={{ fontSize: '0.9rem' }}>ms</span>
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* SAP IconTabBar Inspection Container */}
            <div className="sap-card" style={{ padding: '1.5rem' }}>
              
              {/* SAP Tab Bar Header */}
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  flexWrap: 'wrap',
                  gap: '1rem',
                  borderBottom: '1px solid var(--sap-card-border)',
                  marginBottom: '1.5rem',
                }}
              >
                <div style={{ display: 'flex', gap: '0.25rem', flexWrap: 'wrap' }}>
                  <button
                    onClick={() => setActiveTab('red_marked')}
                    className={`sap-tab-btn ${activeTab === 'red_marked' ? 'active' : ''}`}
                  >
                    <Target size={16} /> Defect Area Localization
                  </button>

                  <button
                    onClick={() => setActiveTab('heatmap')}
                    className={`sap-tab-btn ${activeTab === 'heatmap' ? 'active' : ''}`}
                  >
                    <Activity size={16} /> Thermal Anomaly Heatmap
                  </button>

                  <button
                    onClick={() => setActiveTab('overlay')}
                    className={`sap-tab-btn ${activeTab === 'overlay' ? 'active' : ''}`}
                  >
                    <Layers size={16} /> Blended Overlay View
                  </button>

                  <button
                    onClick={() => setActiveTab('visualization')}
                    className={`sap-tab-btn ${activeTab === 'visualization' ? 'active' : ''}`}
                  >
                    <FileBarChart size={16} /> Multi-Panel Inspection Report
                  </button>
                </div>

                {/* SAP Slider Threshold Control */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', paddingBottom: '0.5rem' }}>
                  <Sliders size={15} color="var(--sap-brand-blue)" />
                  <span style={{ fontSize: '0.8rem', color: 'var(--sap-text-secondary)' }}>
                    Threshold: <strong>{threshold}</strong>
                  </span>
                  <input
                    type="range"
                    min="5.0"
                    max="25.0"
                    step="0.5"
                    value={threshold}
                    onChange={(e) => reRunThreshold(parseFloat(e.target.value))}
                    style={{ width: '110px', cursor: 'pointer' }}
                  />
                </div>
              </div>

              {/* Visual Display Container */}
              <div style={{ position: 'relative', textAlign: 'center', background: '#f8fafc', padding: '1.5rem', borderRadius: '6px', border: '1px solid var(--sap-card-border)' }}>
                <img
                  src={report.images[activeTab] || report.images.red_marked}
                  alt={activeTab}
                  style={{
                    maxHeight: '460px',
                    maxWidth: '100%',
                    borderRadius: '4px',
                    objectFit: 'contain',
                    border: '1px solid #dcdfe4',
                  }}
                />
                
                <button
                  onClick={() => setModalImage(report.images[activeTab] || report.images.red_marked)}
                  style={{
                    position: 'absolute',
                    top: '2rem',
                    right: '2rem',
                    background: '#ffffff',
                    border: '1px solid var(--sap-card-border)',
                    color: 'var(--sap-text-primary)',
                    borderRadius: '4px',
                    padding: '0.4rem',
                    cursor: 'pointer',
                    boxShadow: '0 2px 6px rgba(0, 0, 0, 0.1)',
                  }}
                  title="Expand View"
                >
                  <Maximize2 size={16} />
                </button>
              </div>

              {/* Technical Description Box */}
              <div
                style={{
                  marginTop: '1.25rem',
                  padding: '0.9rem 1.1rem',
                  borderRadius: '6px',
                  background: '#f1f5f9',
                  border: '1px solid #e2e8f0',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem',
                  fontSize: '0.85rem',
                  color: 'var(--sap-text-secondary)',
                }}
              >
                <Eye size={16} color="var(--sap-brand-blue)" />
                {activeTab === 'red_marked' && (
                  <span>
                    <strong>Defect Area Localization:</strong> Highlights anomalous surfaces with red patch fill and bounding box outlines.
                  </span>
                )}
                {activeTab === 'heatmap' && (
                  <span>
                    <strong>Thermal Heatmap:</strong> Renders feature-distance intensity gradients (Blue = Normal, Red = Elevated Deviation).
                  </span>
                )}
                {activeTab === 'overlay' && (
                  <span>
                    <strong>Blended Overlay View:</strong> Blends normalized anomaly distance maps onto the original container image.
                  </span>
                )}
                {activeTab === 'visualization' && (
                  <span>
                    <strong>Multi-Panel Inspection Report:</strong> Side-by-side comparative layout displaying Original Container, Heatmap, and Marked Defect Region.
                  </span>
                )}
              </div>
            </div>

            {/* SAP System Disposition Recommendation Message Strip */}
            <div
              className="sap-card"
              style={{
                padding: '1.25rem 1.5rem',
                borderLeft: report.is_defective ? '5px solid var(--sap-semantic-negative)' : '5px solid var(--sap-semantic-positive)',
              }}
            >
              <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--sap-text-primary)', marginBottom: '0.35rem' }}>
                System Audit Log & Recommended Quality Action
              </h4>
              <p style={{ fontSize: '0.88rem', color: 'var(--sap-text-primary)', lineHeight: 1.5 }}>
                {report.is_defective ? (
                  <>
                    <strong>DISPOSITION ACTION: REJECT CONTAINER.</strong> The PatchCore anomaly detection model identified structural surface deviations with a distance score of <strong>{report.anomaly_score}</strong> (exceeding system threshold {report.threshold}). Route container to quarantine diversion unit.
                  </>
                ) : (
                  <>
                    <strong>DISPOSITION ACTION: RELEASE FOR PACKAGING.</strong> Container visual feature vectors conform strictly to registered normal baseline distribution (distance score <strong>{report.anomaly_score}</strong> below system threshold {report.threshold}). Container cleared for manufacturing pipeline.
                  </>
                )}
              </p>
            </div>
          </div>
        )}
      </main>

      {/* Fullscreen Zoom Modal */}
      {modalImage && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 100,
            background: 'rgba(28, 36, 48, 0.92)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '2rem',
          }}
          onClick={() => setModalImage(null)}
        >
          <button
            onClick={() => setModalImage(null)}
            style={{
              position: 'absolute',
              top: '1.5rem',
              right: '1.5rem',
              background: '#ffffff',
              border: 'none',
              color: 'var(--sap-text-primary)',
              borderRadius: '50%',
              width: '40px',
              height: '40px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 2px 10px rgba(0,0,0,0.2)',
            }}
          >
            <X size={20} />
          </button>
          <img
            src={modalImage}
            alt="Expanded preview"
            style={{
              maxWidth: '90vw',
              maxHeight: '90vh',
              objectFit: 'contain',
              borderRadius: '8px',
              boxShadow: '0 8px 30px rgba(0, 0, 0, 0.3)',
            }}
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </div>
  );
}
