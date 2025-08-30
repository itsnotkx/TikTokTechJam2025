import React, { useEffect, useRef, useState } from 'react';
import { View, StyleSheet, ScrollView, Text, ActivityIndicator } from 'react-native';
import { useVideo } from '../context/VideoContext';
import { CustomVideoPlayer } from '../components/videoPlayer/CustomVideoPlayer';
import { api } from '../api/video';

const ProcessingGate: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { upload, setUpload, setVideoUri } = useVideo();

  // optional fields; backend may not provide them
  const [progress, setProgress] = useState<number | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const backoffRef = useRef<number>(2000); // ms

  const clearTimer = () => {
    if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null; }
  };

  const scheduleNext = () => {
    clearTimer();
    const delay = Math.min(backoffRef.current, 10000);
    timerRef.current = setTimeout(tick, delay);
    backoffRef.current = Math.min(backoffRef.current * 1.5, 10000);
  };

  const tick = async () => {
    // We poll when the job is queued/processing
    if (upload.status !== 'processing') return;
    if (!('jobId' in upload) || !upload.jobId) {
      setUpload({ status: 'error', message: 'Missing job id' });
      clearTimer();
      return;
    }

    try {
      const data = await api.getJob(upload.jobId);
      const s = String(data.status ?? '').toLowerCase();

      // optional fields (ignore if not provided by backend)
      setProgress(typeof (data as any).progress === 'number' ? (data as any).progress : null);
      setMessage(typeof (data as any).message === 'string' ? (data as any).message : null);

      if (s === 'done') {
        const result = (data as any).result ?? data;
        if (result?.resultUrl) setVideoUri(result.resultUrl);
        setUpload({ status: 'done', jobId: upload.jobId, result });
        clearTimer();
        return;
      }

      if (s === 'queued' || s === 'processing') {
        if (upload.status !== s) {
          setUpload({ status: (s as any), jobId: upload.jobId });
        }
        scheduleNext();
        return;
      }

      // unexpected status → treat as error
      setUpload({ status: 'error', message: `Unexpected status: ${s || 'none'}` });
      clearTimer();
    } catch (e: any) {
      // transient issue → keep polling with backoff
      setMessage(e?.message ?? 'Temporary polling issue…');
      scheduleNext();
    }
  };

  useEffect(() => {
    // Start polling when job enters queued/processing
    if (upload.status === 'processing') {
      backoffRef.current = 2000;
      setProgress(null);
      setMessage(null);
      tick();
    } else {
      // for any other state, ensure timers are cleared
      clearTimer();
    }
    return () => clearTimer();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [upload.status]);

  if (upload.status === 'error') {
    return (
      <View style={styles.processingContainer}>
        <Text style={styles.errorTitle}>Processing failed</Text>
        {message ? <Text style={styles.processingHint}>{message}</Text> : null}
        <Text style={styles.processingFootnote}>Please go back and try re-uploading.</Text>
      </View>
    );
  }

  if (upload.status === 'uploading'|| upload.status === 'processing') {
    return (
      <View style={styles.processingContainer}>
        <ActivityIndicator size="large" />
        <Text style={styles.processingTitle}>
          {upload.status === 'uploading' ? 'Uploading…' : 'Processing…'}
        </Text>
        {typeof progress === 'number' && (
          <Text style={styles.processingSub}>
            {`${Math.max(0, Math.min(100, Math.round(progress)))}%`}
          </Text>
        )}
        {!!message && <Text style={styles.processingHint}>{message}</Text>}
        <Text style={styles.processingFootnote}>
          We’ll open the viewer when your video is ready.
        </Text>
      </View>
    );
  }

  // idle/done: show children (the viewer UI)
  return <>{children}</>;
};

const EditorContent: React.FC = () => {
  const { videoUri } = useVideo();

  if (!videoUri) {
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>No video selected</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <ScrollView showsVerticalScrollIndicator={false}>
        <View style={styles.playerContainer}>
          <CustomVideoPlayer videoUri={videoUri} style={styles.videoPlayer} />
        </View>
        <View style={styles.toolsContainer}>
          <Text style={styles.toolsTitle}>Video</Text>
          <Text style={styles.toolsSubtitle}>
            Your processed video will appear above.
          </Text>
        </View>
      </ScrollView>
    </View>
  );
};

export default function Editor() {
  // No EditorProvider needed since trimming/editor state was removed
  return (
    <ProcessingGate>
      <EditorContent />
    </ProcessingGate>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  playerContainer: {
    margin: 20, borderRadius: 12, overflow: 'hidden',
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1, shadowRadius: 8, elevation: 4,
  },
  videoPlayer: { width: '100%', aspectRatio: 9 / 16 },
  toolsContainer: { padding: 20, backgroundColor: '#f8f9fa', margin: 20, borderRadius: 12 },
  toolsTitle: { fontSize: 16, fontWeight: '600', color: '#333', marginBottom: 5 },
  toolsSubtitle: { fontSize: 14, color: '#666' },
  errorText: { textAlign: 'center', fontSize: 16, color: '#666', marginTop: 50 },

  processingContainer: { flex: 1, paddingHorizontal: 24, justifyContent: 'center', alignItems: 'center', backgroundColor: '#fff' },
  processingTitle: { marginTop: 12, fontSize: 18, fontWeight: '600', color: '#333' },
  processingSub: { marginTop: 6, fontSize: 16, color: '#444' },
  processingHint: { marginTop: 8, fontSize: 13, color: '#777', textAlign: 'center' },
  processingFootnote: { marginTop: 14, fontSize: 12, color: '#999', textAlign: 'center' },
  errorTitle: { marginTop: 12, fontSize: 18, fontWeight: '700', color: '#c62828' },
});
