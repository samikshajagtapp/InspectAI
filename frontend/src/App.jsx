import React, { useState, useEffect, useRef } from 'react';
import {
  Upload,
  ShieldCheck,
  AlertOctagon,
  CheckCircle2,
  XCircle,
  Eye,
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
  AlertTriangle,
  ArrowLeft,
  Settings,
  Database,
  Network,
  Pill,
  FlaskConical,
  Syringe,
  Play,
  Bell,
  Trash2
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5001/api';

export default function App() {
    const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [batchStatus, setBatchStatus] = useState({ isRunning: false, total: 0, current: 0, results: [] });
  const [threshold, setThreshold] = useState(10.46);
  const [activeTab, setActiveTab] = useState('red_marked'); // red_marked | heatmap | overlay | visualization
  const [modalImage, setModalImage] = useState(null);
  const [health, setHealth] = useState({ loaded: false, backbone: 'resnet18', accelerator: 'cpu' });

  // 21 CFR Part 11 Override States
  const [overrideApplied, setOverrideApplied] = useState(false);
  const [overrideLog, setOverrideLog] = useState([]);
  const [authUsername, setAuthUsername] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [authReason, setAuthReason] = useState('');

  // Notification States
  const [notifications, setNotifications] = useState([]);
  const [showNotifications, setShowNotifications] = useState(false);

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
          accelerator: data.accelerator || 'cpu',
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

  
  const handleBatchInspect = async () => {
    if (samples.length === 0) return;
    setReport(null);
    setSelectedSample(null);
    setBatchStatus({ isRunning: true, total: samples.length, current: 0, results: [] });

    let currentResults = [];
    // Process in chunks of 8 to speed up batch inspection
    const CHUNK_SIZE = 8;
    for (let i = 0; i < samples.length; i += CHUNK_SIZE) {
      const chunk = samples.slice(i, i + CHUNK_SIZE);
      const promises = chunk.map(async (sample) => {
        try {
          const formData = new FormData();
          formData.append('sample_path', sample.path);
          formData.append('fast', 'true');
          const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            body: formData,
          });
          const data = await response.json();
          return { sample, result: data };
        } catch (err) {
          console.error('Batch error on sample:', sample.id, err);
          return null;
        }
      });
      
      const chunkResults = await Promise.all(promises);
      const validResults = chunkResults.filter(r => r !== null);
      currentResults = [...currentResults, ...validResults];
      setBatchStatus(prev => ({ ...prev, current: Math.min(i + CHUNK_SIZE, samples.length), results: currentResults }));
    }
    setBatchStatus(prev => ({ ...prev, isRunning: false }));
    
    // Add batch notification
    const totalDefects = currentResults.filter(r => r.result.action === 'AUTO-REJECT').length;
    const totalReviews = currentResults.filter(r => r.result.action === 'HUMAN-REVIEW').length;
    if (totalDefects > 0 || totalReviews > 0) {
      setNotifications(prev => [{
        id: Date.now() + Math.random(),
        title: `Batch Inspection Complete`,
        message: `${totalDefects} Rejects, ${totalReviews} Reviews detected out of ${currentResults.length} samples.`,
        type: totalDefects > 0 ? 'reject' : 'review',
        timestamp: new Date().toLocaleTimeString()
      }, ...prev]);
    }
  };

  const handleSampleClick = (sample) => {
    setSelectedSample(sample);
    runInference({ sample_path: sample.path });
  };

  const runInference = async ({ file, sample_path }) => {
    setLoading(true);
    setError(null);
    setReport(null);
    setOverrideApplied(false);
    setOverrideLog([]);

    const formData = new FormData();
    formData.append('threshold', threshold.toString());
    
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

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.message || 'Inspection analysis failed.');
      }

      const data = await res.json();
      setReport(data);

      if (data.action !== 'AUTO-PASS') {
        const title = file ? file.name : (sample_path.split('/').pop() || 'Unknown');
        setNotifications(prev => [{
          id: Date.now() + Math.random(),
          title: `Defect Detected: ${title}`,
          message: `Score: ${data.anomaly_score} | Action: ${data.action}`,
          type: data.action === 'AUTO-REJECT' ? 'reject' : 'review',
          timestamp: new Date().toLocaleTimeString()
        }, ...prev]);
      }
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const reRunThreshold = async (newVal) => {
    setThreshold(newVal);
    if (!report) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('threshold', newVal.toString());

    if (selectedSample) {
      formData.append('sample_path', selectedSample.path);
    } else if (report.filename) {
      setError("Please upload the custom file again to test with the new limit.");
      setLoading(false);
      return;
    } else {
      setLoading(false);
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        setReport(data);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleOverrideSubmit = (e) => {
    e.preventDefault();
    if (!authUsername.trim() || !authPassword.trim() || !authReason.trim()) {
      alert('Please fill in all supervisor authorization fields.');
      return;
    }
    setOverrideLog([
      {
        username: authUsername,
        reason: authReason,
        timestamp: new Date().toLocaleTimeString()
      }
    ]);
    setOverrideApplied(true);
    setAuthUsername('');
    setAuthPassword('');
    setAuthReason('');
  };

  const getSpecs = () => {
    if (!report) return {};
    const action = report.action;
    
    if (overrideApplied) {
      return {
        classId: 'DC-V00 (Overridden)',
        severity: 'NONE (Approved Exception)',
        passFail: 'PASS (Manual override)',
        sapAction: 'QE11 → PASS, QA12 → ACCEPTED (manual override logged)'
      };
    }

    if (action === 'AUTO-PASS') {
      return {
        classId: 'DC-V00 (None)',
        severity: 'NONE',
        passFail: 'PASS',
        sapAction: 'QE11 → PASS, QA12 → ACCEPTED'
      };
    } else if (action === 'HUMAN-REVIEW') {
      return {
        classId: 'DC-V02 (Borderline)',
        severity: 'MINOR (Borderline Grey-Zone)',
        passFail: 'PENDING (Hold)',
        sapAction: 'Route to supervisor queue, lock inspection lot'
      };
    } else {
      const fn = (report.filename || selectedSample?.path || '').toLowerCase();
      const isContamination = fn.includes('contamination') || fn.includes('dirt');
      return {
        classId: isContamination ? 'DC-V01 (Surface Dirt)' : 'DC-V04 (Fractured Casing)',
        severity: 'MAJOR / CRITICAL',
        passFail: 'FAIL',
        sapAction: 'QE11 → FAIL, BAPI_NOTIF_CREATE'
      };
    }
  };

  const specs = getSpecs();

    return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: '#f9fafb', overflow: 'hidden' }}>
      
      {/* Sticky Header - Title is small when report is open */}
      <header style={{
        background: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
        borderBottom: '2.5px solid #000000',
        padding: '0.85rem 1.5rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        position: 'sticky',
        top: 0,
        zIndex: 10,
        height: '70px',
        flexShrink: 0,
        boxShadow: '0 2px 10px rgba(0,0,0,0.1)'
      }}>
        <div 
          style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}
          onClick={() => {
            setReport(null);
            setSelectedSample(null);
          }}
        >
          <div style={{
            width: '34px',
            height: '34px',
            borderRadius: '4px',
            background: 'var(--brand-blue)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}>
            <Activity size={18} color="#ffffff" />
          </div>
          <div>
            {/* Show normal header title only if report is loaded; otherwise it will be highlighted in the main screen */}
            <h1 style={{ fontFamily: 'var(--font-heading)', fontSize: report ? '1.25rem' : '1.05rem', fontWeight: 900, color: '#ffffff', letterSpacing: '-0.02em', display: 'flex', alignItems: 'center', gap: '0.5rem', transition: 'all 0.15s ease' }}>
              INSPECT AI {report && <span style={{ fontSize: '0.58rem', fontWeight: 900, padding: '0.1rem 0.35rem', background: '#10b981', color: '#ffffff', borderRadius: '2px' }}>GMP VALIDATED</span>}
            </h1>
            <span style={{ fontSize: '0.68rem', color: '#cbd5e1', fontFamily: 'var(--font-body)' }}>Automated Bottle Quality Inspection Panel</span>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
          
          {/* Notification Bell */}
          <div style={{ position: 'relative' }}>
            <button 
              onClick={() => setShowNotifications(!showNotifications)}
              style={{
                background: 'transparent',
                border: 'none',
                cursor: 'pointer',
                position: 'relative',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '0.4rem',
                borderRadius: '50%',
                transition: 'background 0.2s',
                backgroundColor: showNotifications ? 'rgba(255,255,255,0.1)' : 'transparent'
              }}
              onMouseOver={e => e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.1)'}
              onMouseOut={e => e.currentTarget.style.backgroundColor = showNotifications ? 'rgba(255,255,255,0.1)' : 'transparent'}
            >
              <Bell size={18} color="#ffffff" />
              {notifications.length > 0 && (
                <div style={{
                  position: 'absolute',
                  top: '0px',
                  right: '0px',
                  background: '#ef4444',
                  color: 'white',
                  fontSize: '0.55rem',
                  fontWeight: 'bold',
                  width: '14px',
                  height: '14px',
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  {notifications.length}
                </div>
              )}
            </button>
            
            {/* Notification Dropdown */}
            {showNotifications && (
              <div style={{
                position: 'absolute',
                top: 'calc(100% + 10px)',
                right: 0,
                width: '320px',
                maxHeight: '400px',
                background: '#ffffff',
                border: '1px solid #e2e8f0',
                borderRadius: '8px',
                boxShadow: '0 10px 25px rgba(0,0,0,0.1)',
                zIndex: 50,
                display: 'flex',
                flexDirection: 'column',
                overflow: 'hidden'
              }}>
                <div style={{ padding: '0.75rem 1rem', borderBottom: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#f8fafc' }}>
                  <span style={{ fontSize: '0.85rem', fontWeight: 800, color: '#1e293b' }}>Alerts & Notifications</span>
                  {notifications.length > 0 && (
                    <button 
                      onClick={() => setNotifications([])}
                      style={{ background: 'none', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.25rem', color: '#64748b', fontSize: '0.7rem' }}
                    >
                      <Trash2 size={12} /> Clear
                    </button>
                  )}
                </div>
                <div style={{ overflowY: 'auto', flex: 1, padding: '0.5rem' }}>
                  {notifications.length === 0 ? (
                    <div style={{ padding: '2rem 1rem', textAlign: 'center', color: '#94a3b8', fontSize: '0.8rem' }}>
                      No new notifications
                    </div>
                  ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                      {notifications.map(n => (
                        <div key={n.id} style={{
                          padding: '0.75rem',
                          borderRadius: '6px',
                          borderLeft: `3px solid ${n.type === 'reject' ? '#ef4444' : '#f59e0b'}`,
                          background: n.type === 'reject' ? '#fef2f2' : '#fffbeb',
                        }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.25rem' }}>
                            <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#1e293b' }}>{n.title}</span>
                            <span style={{ fontSize: '0.6rem', color: '#64748b' }}>{n.timestamp}</span>
                          </div>
                          <div style={{ fontSize: '0.7rem', color: '#475569' }}>{n.message}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.72rem', color: '#10b981', background: '#e6f4ea', padding: '0.25rem 0.55rem', borderRadius: '4px', border: '1px solid #e5e7eb', fontWeight: 'bold' }}>
            <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#10b981' }} />
            <span>Inspection Line: Active</span>
          </div>

          <div style={{
            background: '#f3f4f6',
            border: '1px solid #e5e7eb',
            borderRadius: '4px',
            padding: '0.25rem 0.55rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            fontSize: '0.72rem',
            color: '#6b7280',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              <Cpu size={11} color="#000000" />
              <span>PatchCore ({health.backbone})</span>
            </div>
            <span style={{ color: '#d1d5db' }}>|</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              <Activity size={11} color="#000000" />
              <span>Device: {health.accelerator.toUpperCase()}</span>
            </div>
          </div>
        </div>
      </header>

      {/* 3 Column Grid Layout */}
      <main style={{ width: '100%', maxWidth: '100%', padding: '0.4rem', flex: 1, display: 'flex', gap: '0.4rem', height: 'calc(100vh - 70px)', overflow: 'hidden' }}>
        
        {/* Left Panel - Image Selection */}
        <div style={{ width: '320px', flexShrink: 0, display: 'flex', flexDirection: 'column', height: '100%' }}>
          <div className="glass-panel" style={{ padding: '0.65rem', height: '100%', display: 'flex', flexDirection: 'column', background: '#ffffff', border: '1px solid #e5e7eb' }}>
            <div style={{ marginBottom: '0.5rem' }}>
              <h3 className="section-title" style={{ fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.35rem', color: '#475569', justifyContent: 'space-between' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}><Box size={14} color="#64748b" /> Sample Bottles</span>
              </h3>
            </div>

            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.75rem' }}>
              <button
                onClick={handleBatchInspect}
                disabled={batchStatus.isRunning}
                style={{
                  flex: 1,
                  padding: '0.65rem',
                  background: batchStatus.isRunning ? '#94a3b8' : '#38bdf8',
                  border: 'none',
                  borderRadius: '6px',
                  fontSize: '0.75rem',
                  fontWeight: 800,
                  color: '#ffffff',
                  cursor: batchStatus.isRunning ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '0.4rem',
                  boxShadow: '0 4px 6px rgba(56, 189, 248, 0.25)',
                  transition: 'transform 0.1s ease',
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em'
                }}
                onMouseOver={(e) => !batchStatus.isRunning && (e.currentTarget.style.transform = 'translateY(-1px)')}
                onMouseOut={(e) => !batchStatus.isRunning && (e.currentTarget.style.transform = 'translateY(0)')}
              >
                <Play size={14} fill="#ffffff" /> {batchStatus.isRunning ? 'Inspecting Batch...' : 'Inspect All Images'}
              </button>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
              <div style={{ flex: 1, height: '1px', background: '#e2e8f0' }} />
              <span style={{ fontSize: '0.65rem', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>or choose one below</span>
              <div style={{ flex: 1, height: '1px', background: '#e2e8f0' }} />
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', overflowY: 'auto', flex: 1 }}>
              {samples.map((sample) => {
                const isSelected = selectedSample?.id === sample.id;
                const isGood = sample.category === 'good';
                const isReview = sample.category === 'broken_small';

                let bgColor = isGood ? '#f0fdf4' : (isReview ? '#f8fafc' : '#fef2f2');
                let bgSelected = isGood ? '#dcfce7' : (isReview ? '#f1f5f9' : '#fee2e2');
                let borderColor = isGood ? '#bbf7d0' : (isReview ? '#e2e8f0' : '#fecaca');
                let borderSelected = isGood ? '#4ade80' : (isReview ? '#94a3b8' : '#f87171');
                let borderLeftSelected = isGood ? '#16a34a' : (isReview ? '#64748b' : '#dc2626');
                
                let statusColor = isGood ? 'var(--status-pass)' : (isReview ? '#475569' : 'var(--status-reject)');
                let statusBg = isGood ? 'var(--status-pass-bg)' : (isReview ? '#e2e8f0' : 'var(--status-reject-bg)');
                let statusText = isGood ? 'Conforming' : (isReview ? 'Review' : 'Defect');

                return (
                  <div
                    key={sample.id}
                    onClick={() => handleSampleClick(sample)}
                    className={`glass-panel glass-panel-hover`}
                    style={{
                      padding: '0.35rem',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.45rem',
                      borderRadius: '6px',
                      background: isSelected ? bgSelected : bgColor,
                      border: isSelected ? `1px solid ${borderSelected}` : `1px solid ${borderColor}`,
                      borderLeft: isSelected ? `4px solid ${borderLeftSelected}` : `1px solid ${borderColor}`,
                      boxShadow: isSelected ? '0 2px 5px rgba(0,0,0,0.05)' : 'none',
                      transition: 'all 0.15s ease'
                    }}
                  >
                    {sample.image_b64 ? (
                      <img
                        src={sample.image_b64}
                        alt={sample.title}
                        style={{ width: '32px', height: '32px', objectFit: 'cover', borderRadius: '3px', border: '1px solid #e5e7eb' }}
                      />
                    ) : (
                      <div style={{ width: '32px', height: '32px', background: '#f9fafb', borderRadius: '3px' }} />
                    )}
                    <div style={{ minWidth: 0, flex: 1 }}>
                      <h4 style={{ fontSize: '0.7rem', fontWeight: 800, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {sample.title}
                      </h4>
                      <span style={{
                        fontSize: '0.58rem',
                        fontWeight: 700,
                        color: statusColor,
                        padding: '0.05rem 0.25rem',
                        background: statusBg,
                        borderRadius: '2px',
                        display: 'inline-block',
                        marginTop: '0.1rem'
                      }}>
                        {statusText}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* COLUMN 2: CENTER PANEL (flex: 1, full height) */}
        <div style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          gap: '0.4rem',
          height: '100%',
          minWidth: 0,
          background: 'linear-gradient(180deg, #d2dbe5 0%, #ffffff 100%)',
          border: '1px solid #e5e7eb',
          borderRadius: '4px',
          padding: '0.65rem'
        }}>
          
          {/* Default Awaiting Scan State - High-end Professional Enterprise UI */}
          {!report && !loading && !batchStatus.isRunning && batchStatus.results.length === 0 && (
            <div style={{ 
              display: 'flex', 
              flexDirection: 'column', 
              justifyContent: 'center', 
              alignItems: 'center', 
              flex: 1, 
              position: 'relative',
              overflow: 'hidden',
              background: '#ffffff',
              borderRadius: '6px',
              border: '1px solid #e2e8f0'
            }}>
              {/* Sophisticated Background Graphics */}
              <div style={{ position: 'absolute', inset: 0, backgroundImage: 'linear-gradient(rgba(226, 232, 240, 0.4) 1px, transparent 1px), linear-gradient(90deg, rgba(226, 232, 240, 0.4) 1px, transparent 1px)', backgroundSize: '40px 40px', zIndex: 0 }} />
              <div style={{ position: 'absolute', top: 0, bottom: 0, left: '50%', width: '1px', background: 'linear-gradient(180deg, transparent, rgba(16,185,129,0.3), transparent)', zIndex: 0 }} />
              <div style={{ position: 'absolute', top: '50%', left: 0, right: 0, height: '1px', background: 'linear-gradient(90deg, transparent, rgba(16,185,129,0.3), transparent)', zIndex: 0 }} />
              
              {/* Corner targeting brackets */}
              <div style={{ position: 'absolute', top: '2rem', left: '2rem', width: '20px', height: '20px', borderTop: '2px solid #94a3b8', borderLeft: '2px solid #94a3b8', zIndex: 0 }} />
              <div style={{ position: 'absolute', top: '2rem', right: '2rem', width: '20px', height: '20px', borderTop: '2px solid #94a3b8', borderRight: '2px solid #94a3b8', zIndex: 0 }} />
              <div style={{ position: 'absolute', bottom: '2rem', left: '2rem', width: '20px', height: '20px', borderBottom: '2px solid #94a3b8', borderLeft: '2px solid #94a3b8', zIndex: 0 }} />
              <div style={{ position: 'absolute', bottom: '2rem', right: '2rem', width: '20px', height: '20px', borderBottom: '2px solid #94a3b8', borderRight: '2px solid #94a3b8', zIndex: 0 }} />

              <div style={{ 
                position: 'relative', 
                zIndex: 1, 
                display: 'flex', 
                flexDirection: 'column', 
                alignItems: 'center',
                background: 'rgba(255, 255, 255, 0.85)',
                backdropFilter: 'blur(16px)',
                padding: '3rem 4rem',
                borderRadius: '16px',
                border: '1px solid #38bdf8',
                boxShadow: '0 20px 40px rgba(0,0,0,0.03), 0 1px 3px rgba(0,0,0,0.05)'
              }}>
                <div style={{
                  width: '64px',
                  height: '64px',
                  borderRadius: '16px',
                  background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
                  border: '1px solid rgba(56, 189, 248, 0.5)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginBottom: '1.5rem',
                  boxShadow: '0 10px 15px -3px rgba(15, 23, 42, 0.2)'
                }}>
                  <Activity size={30} color="#38bdf8" strokeWidth={2} />
                </div>

                <h2 style={{ 
                  fontFamily: 'var(--font-heading)', 
                  fontSize: '2.5rem', 
                  fontWeight: 800, 
                  color: '#0f172a',
                  letterSpacing: '-0.02em',
                  margin: 0,
                  lineHeight: 1
                }}>
                  INSPECT AI
                </h2>
                <div style={{ width: '40px', height: '3px', background: '#38bdf8', borderRadius: '2px', margin: '1.25rem 0' }} />
                
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.5rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <span style={{ display: 'inline-block', width: '8px', height: '8px', borderRadius: '50%', background: '#10b981', boxShadow: '0 0 10px rgba(16,185,129,0.5)' }}></span>
                    <span style={{ fontSize: '0.75rem', color: '#475569', fontWeight: 700, letterSpacing: '0.05em', textTransform: 'uppercase' }}>System Ready</span>
                  </div>
                  <div style={{ width: '1px', height: '12px', background: '#cbd5e1' }} />
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <Cpu size={14} color="#475569" />
                    <span style={{ fontSize: '0.75rem', color: '#475569', fontWeight: 700, letterSpacing: '0.05em', textTransform: 'uppercase' }}>Model Active</span>
                  </div>
                </div>

                <p style={{ fontSize: '0.9rem', color: '#64748b', textAlign: 'center', maxWidth: '320px', lineHeight: 1.6, margin: 0 }}>
                  Awaiting sensor input. Select a sample from the inspection queue to begin visual analysis.
                </p>
              </div>
            </div>
          )}

          
          {/* Batch Inspection View */}
          {!report && (batchStatus.isRunning || batchStatus.results.length > 0) && (
            <div style={{ padding: '2rem', flex: 1, display: 'flex', flexDirection: 'column', overflowY: 'auto' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '1rem' }}>
                <div>
                  <h2 style={{ fontSize: '1.75rem', fontWeight: 800, color: '#1e293b', margin: 0 }}>Batch Inspection Results</h2>
                  <p style={{ fontSize: '0.9rem', color: '#64748b', margin: '0.2rem 0 0 0' }}>Processed {batchStatus.results.length} of {batchStatus.total} samples</p>
                </div>
                {!batchStatus.isRunning && (
                  <button 
                    onClick={() => setBatchStatus({ isRunning: false, total: 0, current: 0, results: [] })}
                    style={{ padding: '0.5rem 1rem', background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '4px', cursor: 'pointer', fontSize: '0.75rem', fontWeight: 600, color: '#475569' }}
                  >
                    Clear Batch
                  </button>
                )}
              </div>
              
              {batchStatus.isRunning && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.5rem', background: '#f8fafc', padding: '1rem', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                  <div style={{ height: '8px', flex: 1, background: '#e2e8f0', borderRadius: '4px', overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${(batchStatus.current / batchStatus.total) * 100}%`, background: 'var(--brand-blue)', transition: 'width 0.2s ease' }} />
                  </div>
                  <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#64748b' }}>
                    {batchStatus.current} / {batchStatus.total}
                  </span>
                </div>
              )}
              
              {!batchStatus.isRunning && batchStatus.results.length > 0 && (() => {
                const good = batchStatus.results.filter(r => r.result.action === 'AUTO-PASS').length;
                const review = batchStatus.results.filter(r => r.result.action === 'HUMAN-REVIEW').length;
                const defect = batchStatus.results.filter(r => r.result.action === 'AUTO-REJECT').length;
                return (
                  <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem' }}>
                    <div style={{ flex: 1, background: '#f0fdf4', border: '1px solid #bbf7d0', padding: '1rem', borderRadius: '6px' }}>
                      <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#16a34a', textTransform: 'uppercase' }}>Good / Conforming</div>
                      <div style={{ fontSize: '1.5rem', fontWeight: 900, color: '#15803d' }}>{good}</div>
                    </div>
                    <div style={{ flex: 1, background: '#f8fafc', border: '1px solid #e2e8f0', padding: '1rem', borderRadius: '6px' }}>
                      <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#64748b', textTransform: 'uppercase' }}>Human Review Needed</div>
                      <div style={{ fontSize: '1.5rem', fontWeight: 900, color: '#475569' }}>{review}</div>
                    </div>
                    <div style={{ flex: 1, background: '#fef2f2', border: '1px solid #fecaca', padding: '1rem', borderRadius: '6px' }}>
                      <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#dc2626', textTransform: 'uppercase' }}>Defected</div>
                      <div style={{ fontSize: '1.5rem', fontWeight: 900, color: '#b91c1c' }}>{defect}</div>
                    </div>
                  </div>
                )
              })()}

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '1rem' }}>
                {batchStatus.results.map((r, idx) => {
                  const isPass = r.result.action === 'AUTO-PASS';
                  const isReview = r.result.action === 'HUMAN-REVIEW';
                  const isFail = r.result.action === 'AUTO-REJECT';

                  const cardBg = isPass ? '#f0fdf4' : (isReview ? '#f8fafc' : '#fef2f2');
                  const cardBorder = isPass ? '#10b981' : (isReview ? '#94a3b8' : '#ef4444');
                  const cardTextColor = isPass ? '#047857' : (isReview ? '#475569' : '#b91c1c');
                  const cardLabel = isPass ? 'PASS' : (isReview ? 'REVIEW' : 'FAIL');

                  return (
                    <div key={idx} onClick={() => handleSampleClick(r.sample)} style={{ border: `1px solid ${cardBorder}`, borderRadius: '6px', padding: '0.5rem', cursor: 'pointer', background: cardBg, display: 'flex', flexDirection: 'column', boxShadow: '0 1px 3px rgba(0,0,0,0.05)', transition: 'transform 0.1s ease' }} onMouseOver={(e) => e.currentTarget.style.transform = 'translateY(-2px)'} onMouseOut={(e) => e.currentTarget.style.transform = 'translateY(0)'}>
                      <img src={r.sample.image_b64} alt={r.sample.title} style={{ width: '100%', aspectRatio: '1', objectFit: 'cover', borderRadius: '4px', marginBottom: '0.5rem', border: '1px solid rgba(0,0,0,0.1)' }} />
                      <div style={{ fontSize: '0.75rem', fontWeight: 800, color: cardTextColor, marginBottom: '0.2rem' }}>
                        {cardLabel}
                      </div>
                      <div style={{ fontSize: '0.65rem', color: '#475569', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', fontWeight: 500 }}>
                        {r.sample.title}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Loader State */}
          {loading && (
            <div className="glass-panel" style={{ padding: '3rem 1.5rem', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem', flex: 1, border: '1px solid #e5e7eb' }}>
              <div className="modern-spinner" />
              <div>
                <h3 style={{ fontFamily: 'var(--font-heading)', fontSize: '1rem', fontWeight: 900, color: 'var(--text-primary)' }}>
                  Scanning Image...
                </h3>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.72rem', marginTop: '0.2rem' }}>
                  Analyzing bottle surface for cosmetic and structural defects
                </p>
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div
              style={{
                borderRadius: '4px',
                padding: '0.65rem 0.85rem',
                background: 'var(--status-reject-bg)',
                border: '1px solid #e5e7eb',
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                flexShrink: 0
              }}
            >
              <AlertOctagon size={15} color="var(--status-reject)" />
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 800, color: 'var(--text-primary)' }}>Scanner Error</span>
                <span style={{ fontSize: '0.68rem', color: 'var(--text-secondary)' }}>{error}</span>
              </div>
            </div>
          )}

          {/* Scanned Inspection details display */}
          {report && !loading && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', flex: 1, minHeight: 0, overflowY: 'auto' }}>
              
              {/* Back / Reset button */}
              <div style={{ display: 'flex', marginBottom: '0.2rem' }}>
                <button
                  onClick={() => {
                    setReport(null);
                    setSelectedSample(null);
                  }}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    background: '#ffffff',
                    border: '1px solid #e2e8f0',
                    padding: '0.4rem 0.8rem',
                    borderRadius: '4px',
                    fontSize: '0.75rem',
                    fontWeight: 700,
                    color: '#475569',
                    cursor: 'pointer',
                    boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
                    transition: 'all 0.15s ease'
                  }}
                >
                  <ArrowLeft size={14} /> Back to Selection
                </button>
              </div>

              {/* Verdict Header highlight strip */}
              {(() => {
                let badgeColor = 'var(--status-pass)';
                let bgStyle = 'var(--status-pass-bg)';
                let borderStyle = '2.5px solid #000000';
                let label = 'AUTO-PASS';
                let desc = 'APPROVED — Bottle meets all quality inspection metrics';
                let icon = <CheckCircle2 size={22} color="var(--status-pass)" />;

                if (overrideApplied) {
                  badgeColor = 'var(--brand-blue)';
                  bgStyle = 'var(--brand-blue-light)';
                  borderStyle = '2.5px solid #000000';
                  label = 'APPROVED (OVERRIDE)';
                  desc = 'MANUAL OVERRIDE APPROVED — Released to production line';
                  icon = <ShieldCheck size={22} color="var(--brand-blue)" />;
                } else if (report.action === 'HUMAN-REVIEW') {
                  badgeColor = 'var(--status-review)';
                  bgStyle = 'var(--status-review-bg)';
                  borderStyle = '2.5px solid #000000';
                  label = 'HUMAN-REVIEW';
                  desc = 'BORDERLINE LIMITS — Suspended pending supervisor sign-off';
                  icon = <AlertTriangle size={22} color="var(--status-review)" />;
                } else if (report.action === 'AUTO-REJECT') {
                  badgeColor = 'var(--status-reject)';
                  bgStyle = 'var(--status-reject-bg)';
                  borderStyle = '2.5px solid #000000';
                  label = 'AUTO-REJECT';
                  desc = 'REJECTED — Container visual defects exceed allowed limit';
                  icon = <XCircle size={22} color="var(--status-reject)" />;
                }

                return (
                  <div
                    className="glass-panel"
                    style={{
                      padding: '0.85rem 1.25rem',
                      background: bgStyle,
                      border: borderStyle,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      flexWrap: 'wrap',
                      gap: '0.5rem',
                      flexShrink: 0
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                      <div style={{
                        width: '38px',
                        height: '38px',
                        borderRadius: '4px',
                        background: '#ffffff',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        border: '1px solid #e5e7eb'
                      }}>
                        {icon}
                      </div>
                      <div>
                        <span style={{ fontSize: '0.65rem', fontWeight: 900, color: badgeColor, textTransform: 'uppercase', letterSpacing: '0.08em', display: 'block' }}>
                          Verdict: {label}
                        </span>
                        <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '0.95rem', fontWeight: 900, color: badgeColor, marginTop: '0.1rem' }}>
                          {desc}
                        </h2>
                      </div>
                    </div>

                    {/* Stats Strip */}
                    <div style={{ display: 'flex', gap: '1.25rem', alignItems: 'center' }}>
                      <div>
                        <span style={{ fontSize: '0.58rem', color: '#000000', display: 'block', textTransform: 'uppercase', fontWeight: 800 }}>AI Score</span>
                        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.95rem', fontWeight: 900, color: badgeColor }}>{report.anomaly_score.toFixed(2)}</span>
                      </div>
                      <div style={{ width: '1px', height: '20px', background: '#000000' }} />
                      <div>
                        <span style={{ fontSize: '0.58rem', color: '#000000', display: 'block', textTransform: 'uppercase', fontWeight: 800 }}>Standard Limit</span>
                        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.95rem', fontWeight: 900, color: 'var(--text-primary)' }}>{report.threshold.toFixed(2)}</span>
                      </div>
                      <div style={{ width: '1px', height: '20px', background: '#000000' }} />
                      <div>
                        <span style={{ fontSize: '0.58rem', color: '#000000', display: 'block', textTransform: 'uppercase', fontWeight: 800 }}>Scan Time</span>
                        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.95rem', fontWeight: 900, color: '#000000' }}>{report.latency_ms} <span style={{ fontSize: '0.65rem' }}>ms</span></span>
                      </div>
                    </div>
                  </div>
                );
              })()}

              {/* Main Visualizer Image Display Card */}
              <div className="glass-panel" style={{ padding: '0.5rem', display: 'flex', flexDirection: 'column', gap: '0.35rem', background: '#ffffff', flexShrink: 0, border: '1px solid #e5e7eb' }}>
                
                {/* Visualizer selector tab header */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid #e5e7eb', paddingBottom: '0.25rem', gap: '0.4rem', flexShrink: 0 }}>
                  <div style={{ display: 'flex', gap: '0.1rem' }}>
                    <button
                      onClick={() => setActiveTab('red_marked')}
                      className={`tab-btn ${activeTab === 'red_marked' ? 'active' : ''}`}
                      style={{ padding: '0.3rem 0.5rem', fontSize: '0.72rem', color: '#000000' }}
                    >
                      <Target size={11} /> BBox
                    </button>
                    <button
                      onClick={() => setActiveTab('heatmap')}
                      className={`tab-btn ${activeTab === 'heatmap' ? 'active' : ''}`}
                      style={{ padding: '0.3rem 0.5rem', fontSize: '0.72rem', color: '#000000' }}
                    >
                      <Activity size={11} /> Heatmap
                    </button>
                    <button
                      onClick={() => setActiveTab('overlay')}
                      className={`tab-btn ${activeTab === 'overlay' ? 'active' : ''}`}
                      style={{ padding: '0.3rem 0.5rem', fontSize: '0.72rem', color: '#000000' }}
                    >
                      <Layers size={11} /> Blended
                    </button>
                    <button
                      onClick={() => setActiveTab('visualization')}
                      className={`tab-btn ${activeTab === 'visualization' ? 'active' : ''}`}
                      style={{ padding: '0.3rem 0.5rem', fontSize: '0.72rem', color: '#000000' }}
                    >
                      <FileBarChart size={11} /> Multi-Panel
                    </button>
                  </div>

                  {/* Sensitivity Range Selector */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                    <Sliders size={11} color="#000000" />
                    <span style={{ fontSize: '0.62rem', color: '#000000', fontWeight: 800 }}>
                      Sens: <strong>{threshold}</strong>
                    </span>
                    <input
                      type="range"
                      min="5.0"
                      max="25.0"
                      step="0.5"
                      value={threshold}
                      onChange={(e) => reRunThreshold(parseFloat(e.target.value))}
                      style={{ width: '60px' }}
                    />
                  </div>
                </div>

                {/* Container Image Display */}
                <div style={{
                  position: 'relative',
                  background: '#f3f4f6',
                  padding: '0.2rem',
                  borderRadius: '3px',
                  border: '1px solid #e5e7eb',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: '220px',
                  flexShrink: 0
                }}>
                  <img
                    src={report.images[activeTab] || report.images.red_marked}
                    alt={activeTab}
                    style={{
                      maxHeight: '100%',
                      maxWidth: '100%',
                      borderRadius: '3px',
                      objectFit: 'contain',
                      border: '1px solid #e5e7eb'
                    }}
                  />
                  <button
                    onClick={() => setModalImage(report.images[activeTab] || report.images.red_marked)}
                    style={{
                      position: 'absolute',
                      top: '0.4rem',
                      right: '0.4rem',
                      background: '#ffffff',
                      border: '1px solid #e5e7eb',
                      color: '#000000',
                      borderRadius: '3px',
                      padding: '0.2rem',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                    title="Fullscreen zoom"
                  >
                    <Maximize2 size={11} />
                  </button>
                </div>

                {/* Subtitle Caption */}
                <div style={{ fontSize: '0.65rem', color: '#000000', background: '#f3f4f6', padding: '0.25rem 0.45rem', borderRadius: '3px', border: '1px solid #e5e7eb', display: 'flex', alignItems: 'center', gap: '0.35rem', flexShrink: 0 }}>
                  <Eye size={11} color="var(--brand-blue)" />
                  <span>Highlights defect locations on the bottle image above.</span>
                </div>
              </div>

              {/* INSPECTION DETAILS Specs Table Card */}
              <div className="glass-panel" style={{ padding: '0.85rem', background: '#ffffff', flex: 1, minHeight: 0, border: '1px solid #e5e7eb' }}>
                <h3 className="section-title" style={{ fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '0.35rem', borderBottom: '1px solid #e5e7eb', paddingBottom: '0.4rem', marginBottom: '0.25rem', color: '#000000' }}>
                  <FileText size={14} color="#000000" /> Inspection Details
                </h3>

                {/* Grid Border Table */}
                <table style={{ width: '100%', borderCollapse: 'collapse', border: '1px solid #e5e7eb', background: '#ffffff' }}>
                  <thead>
                    <tr style={{ background: '#f3f4f6', borderBottom: '1px solid #e5e7eb' }}>
                      <th style={{ borderRight: '1px solid #e5e7eb', padding: '0.45rem', fontSize: '0.75rem', fontWeight: 900, textAlign: 'left', textTransform: 'uppercase', color: '#000000' }}>Field</th>
                      <th style={{ padding: '0.45rem', fontSize: '0.75rem', fontWeight: 900, textAlign: 'left', textTransform: 'uppercase', color: '#000000' }}>Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr style={{ borderBottom: '1px solid #e5e7eb' }}>
                      <td style={{ borderRight: '1px solid #e5e7eb', padding: '0.45rem', fontSize: '0.85rem', fontWeight: 800, color: '#000000' }}>Defect Class ID</td>
                      <td style={{ padding: '0.45rem', fontFamily: 'var(--font-mono)', fontSize: '0.85rem', fontWeight: 'bold', color: '#000000' }}>{specs.classId}</td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid #e5e7eb' }}>
                      <td style={{ borderRight: '1px solid #e5e7eb', padding: '0.45rem', fontSize: '0.85rem', fontWeight: 800, color: '#000000' }}>Severity</td>
                      <td style={{ padding: '0.45rem', fontSize: '0.85rem', fontWeight: 'bold', textTransform: 'uppercase', color: '#000000' }}>{specs.severity}</td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid #e5e7eb' }}>
                      <td style={{ borderRight: '1px solid #e5e7eb', padding: '0.45rem', fontSize: '0.85rem', fontWeight: 800, color: '#000000' }}>Pass / Fail</td>
                      <td style={{
                        padding: '0.45rem',
                        fontSize: '0.85rem',
                        fontWeight: 'bold',
                        color: specs.passFail.includes('FAIL') ? 'var(--status-reject)' : specs.passFail.includes('PASS') ? 'var(--status-pass)' : 'var(--status-review)',
                        textTransform: 'uppercase'
                      }}>{specs.passFail}</td>
                    </tr>
                    <tr>
                      <td style={{ borderRight: '1px solid #e5e7eb', padding: '0.45rem', fontSize: '0.85rem', fontWeight: 800, color: '#000000' }}>SAP Action</td>
                      <td style={{ padding: '0.45rem' }}>
                        <code style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem', color: '#000000', display: 'block', wordBreak: 'break-word', lineHeight: '1.35', fontWeight: 'bold' }}>
                          {specs.sapAction}
                        </code>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

            </div>
          )}

        </div>

        {/* COLUMN 3: RIGHT PANEL (340px fixed, full height) - Occupied entirely with the Exception Override Box */}
        <div style={{ width: '340px', flexShrink: 0, display: 'flex', flexDirection: 'column', gap: '0.4rem', height: '100%' }}>
          
          <div className="glass-panel" style={{
            padding: '0.85rem',
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'start',
            background: report && report.action === 'HUMAN-REVIEW' && !overrideApplied ? 'var(--status-review-bg)' : '#ffffff',
            border: '1px solid #e5e7eb',
            overflowY: 'auto'
          }}>
            <h3 className="section-title" style={{ fontSize: '0.95rem', display: 'flex', alignItems: 'center', gap: '0.35rem', borderBottom: '1px solid #e5e7eb', paddingBottom: '0.4rem', marginBottom: '0.75rem', color: '#000000' }}>
              <ShieldCheck size={14} color="var(--brand-blue)" /> Exception Override
            </h3>
            
            {!report ? (
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center', padding: '1.5rem' }}>
                <p style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
                  Awaiting bottle scan to activate supervisor exception override controls.
                </p>
              </div>
            ) : overrideApplied ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', background: 'var(--status-pass-bg)', border: '1px solid #e5e7eb', padding: '0.75rem', borderRadius: '4px', flex: 1, justifyContent: 'center' }}>
                <span style={{ fontFamily: 'var(--font-heading)', fontSize: '0.95rem', fontWeight: 900, color: 'var(--status-pass)', display: 'block' }}>✓ OVERRIDE APPROVED</span>
                <p style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', lineHeight: '1.3' }}>
                  E-Signature verified. Exception decision has been securely logged to SAP QM.
                </p>
                <div style={{ borderTop: '1px dashed #e5e7eb', paddingTop: '0.35rem', fontSize: '0.68rem', color: 'var(--text-muted)' }}>
                  Approved by: <strong>{overrideLog[0]?.username}</strong><br/>
                  Reason: <em>"{overrideLog[0]?.reason}"</em><br/>
                  Timestamp: {overrideLog[0]?.timestamp}
                </div>
              </div>
            ) : report.action === 'HUMAN-REVIEW' ? (
              <form onSubmit={handleOverrideSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem', flex: 1 }}>
                <p style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', lineHeight: '1.3' }}>
                  Lot is in grey-zone. Enter supervisor credentials to manually sign and release.
                </p>
                
                <div>
                  <label style={{ fontSize: '0.62rem', fontWeight: 800, color: 'var(--text-secondary)', display: 'block', marginBottom: '0.2rem', textTransform: 'uppercase' }}>
                    Supervisor User
                  </label>
                  <input
                    type="text"
                    required
                    placeholder="Username"
                    value={authUsername}
                    onChange={(e) => setAuthUsername(e.target.value)}
                    style={{ width: '100%', padding: '0.35rem 0.5rem', borderRadius: '3px', border: '1px solid #e5e7eb', fontSize: '0.78rem' }}
                  />
                </div>

                <div>
                  <label style={{ fontSize: '0.62rem', fontWeight: 800, color: 'var(--text-secondary)', display: 'block', marginBottom: '0.2rem', textTransform: 'uppercase' }}>
                    PIN / Password
                  </label>
                  <input
                    type="password"
                    required
                    placeholder="••••••••"
                    value={authPassword}
                    onChange={(e) => setAuthPassword(e.target.value)}
                    style={{ width: '100%', padding: '0.35rem 0.5rem', borderRadius: '3px', border: '1px solid #e5e7eb', fontSize: '0.78rem' }}
                  />
                </div>

                <div>
                  <label style={{ fontSize: '0.62rem', fontWeight: 800, color: 'var(--text-secondary)', display: 'block', marginBottom: '0.2rem', textTransform: 'uppercase' }}>
                    Exception Reason
                  </label>
                  <textarea
                    required
                    rows={3}
                    placeholder="Cosmetic scratch only, outer shell integrity OK."
                    value={authReason}
                    onChange={(e) => setAuthReason(e.target.value)}
                    style={{ width: '100%', padding: '0.35rem 0.5rem', borderRadius: '3px', border: '1px solid #e5e7eb', fontSize: '0.75rem', resize: 'vertical' }}
                  />
                </div>

                <button
                  type="submit"
                  style={{
                    background: 'var(--brand-blue)',
                    color: '#ffffff',
                    fontWeight: 800,
                    border: 'none',
                    padding: '0.45rem',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '0.75rem',
                    marginTop: '0.25rem',
                    width: '100%',
                  }}
                >
                  Sign & Approve Lot (21 CFR Part 11)
                </button>
              </form>
            ) : (
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center', padding: '1.5rem' }}>
                <p style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
                  Standard automatic evaluation mode. No supervisor override signature required.
                </p>
              </div>
            )}
          </div>

        </div>

      </main>

      {/* Lightbox Zoom Modal */}
      {modalImage && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 100,
            background: 'rgba(10, 15, 26, 0.85)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '2rem',
            backdropFilter: 'blur(3px)'
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
              border: '1px solid #e5e7eb',
              color: '#000000',
              borderRadius: '50%',
              width: '36px',
              height: '36px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <X size={18} />
          </button>
          <img
            src={modalImage}
            alt="Inspection Zoom"
            style={{
              maxWidth: '90vw',
              maxHeight: '90vh',
              objectFit: 'contain',
              borderRadius: '4px',
              boxShadow: '0 4px 20px rgba(0,0,0,0.15)',
              border: '1px solid #e5e7eb'
            }}
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </div>
  );
}
