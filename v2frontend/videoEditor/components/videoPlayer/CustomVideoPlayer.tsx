import React, { useRef, useEffect, useState } from 'react';
import { View, StyleSheet } from 'react-native';
import { Video, AVPlaybackStatus } from 'expo-av';
import { useEditor } from '../../context/EditorContext';

interface CustomVideoPlayerProps {
  videoUri: string;
  style?: any;
}

export const CustomVideoPlayer: React.FC<CustomVideoPlayerProps> = ({
  videoUri,
  style
}) => {
  const videoRef = useRef<Video>(null);
  const [status, setStatus] = useState<AVPlaybackStatus | null>(null);
  const { 
    currentTime, 
    isPlaying, 
    setVideoDuration, 
    setCurrentTime,
    trimStart,
    trimEnd
  } = useEditor();

  // Seek when currentTime changes from timeline
  useEffect(() => {
    if (videoRef.current && status && 'positionMillis' in status) {
      const currentTimeMs = currentTime * 1000;
      const statusTimeMs = status.positionMillis || 0;
      
      // Only seek if there's a significant difference (avoid infinite loops)
      if (Math.abs(currentTimeMs - statusTimeMs) > 200) {
        videoRef.current.setPositionAsync(currentTimeMs);
      }
    }
  }, [currentTime]);

  // Control playback state
  useEffect(() => {
    if (videoRef.current && status && 'isLoaded' in status && status.isLoaded) {
      if (isPlaying && !status.isPlaying) {
        videoRef.current.playAsync();
      } else if (!isPlaying && status.isPlaying) {
        videoRef.current.pauseAsync();
      }
    }
  }, [isPlaying, status]);

  const onPlaybackStatusUpdate = (status: AVPlaybackStatus) => {
    setStatus(status);
    
    if ('isLoaded' in status && status.isLoaded) {
      // Set video duration when loaded
      if (status.durationMillis) {
        const durationSeconds = status.durationMillis / 1000;
        setVideoDuration(durationSeconds);
      }
      
      // Update current time
      if (status.positionMillis !== undefined) {
        const currentSeconds = status.positionMillis / 1000;
        
        // Respect trim boundaries
        if (trimEnd > 0 && currentSeconds >= trimEnd) {
          videoRef.current?.pauseAsync();
          videoRef.current?.setPositionAsync(trimStart * 1000);
          return;
        }
        
        setCurrentTime(currentSeconds);
      }
    }
  };

  return (
    <View style={[styles.container, style]}>
      <Video
        ref={videoRef}
        source={{ uri: videoUri }}
        style={styles.video}
        resizeMode="contain"
        shouldPlay={isPlaying}
        isLooping={false}
        onPlaybackStatusUpdate={onPlaybackStatusUpdate}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#000',
    borderRadius: 8,
    overflow: 'hidden',
  },
  video: {
    width: '100%',
    height: '100%',
  },
});