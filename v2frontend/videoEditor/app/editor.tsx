import React, { useEffect } from 'react';
import { View, StyleSheet, ScrollView, Text } from 'react-native';
import { useVideo } from '../context/VideoContext';
import { EditorProvider, useEditor } from '../context/EditorContext';
import { CustomVideoPlayer } from '../components/videoPlayer/CustomVideoPlayer';
import { VideoTimeline } from '../components/timeline/VideoTimeline';
import { EditorControls } from '../components/editor/EditorControls';

const EditorContent: React.FC = () => {
  const { videoUri } = useVideo();
  const { setTrimPoints, videoDuration } = useEditor();

  // Initialize trim points when video loads
  useEffect(() => {
    if (videoDuration > 0) {
      setTrimPoints(0, videoDuration);
    }
  }, [videoDuration]);

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
        {/* Video Player */}
        <View style={styles.playerContainer}>
          <CustomVideoPlayer 
            videoUri={videoUri}
            style={styles.videoPlayer}
          />
        </View>

        {/* Timeline and Trim Tool */}
        <VideoTimeline />
        
        {/* Additional editing tools can be added here */}
        <View style={styles.toolsContainer}>
          <Text style={styles.toolsTitle}>Editing Tools</Text>
          <Text style={styles.toolsSubtitle}>
            Use the timeline above to trim your video. More tools coming soon!
          </Text>
        </View>
      </ScrollView>

      {/* Controls fixed at bottom*/}
      <EditorControls />
    </View>
  );
};

export default function Editor() {
  return (
    <EditorProvider>
      <EditorContent />
    </EditorProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  playerContainer: {
    margin: 20,
    borderRadius: 12,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  videoPlayer: {
    width: '100%',
    aspectRatio: 16/9,
  },
  toolsContainer: {
    padding: 20,
    backgroundColor: '#f8f9fa',
    margin: 20,
    borderRadius: 12,
  },
  toolsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 5,
  },
  toolsSubtitle: {
    fontSize: 14,
    color: '#666',
  },
  errorText: {
    textAlign: 'center',
    fontSize: 16,
    color: '#666',
    marginTop: 50,
  },
});