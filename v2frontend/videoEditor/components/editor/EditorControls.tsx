import React, { useState } from 'react';
import { View, StyleSheet, TouchableOpacity, Text, Alert } from 'react-native';
import { useEditor } from '../../context/EditorContext';
import { useVideo } from '../../context/VideoContext';
import { ffmpegService } from '../../services/FFmpegService';

export const EditorControls: React.FC = () => {
  const { isPlaying, setIsPlaying, trimStart, trimEnd, clearEdits } = useEditor();
  const { videoUri, setVideoUri } = useVideo();
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);

  const togglePlayback = () => {
    setIsPlaying(!isPlaying);
  };

  const handleTrimVideo = async () => {
    if (!videoUri) return;

    try {
      setIsProcessing(true);
      
      const trimmedUri = await ffmpegService.trimVideo(
        videoUri,
        trimStart,
        trimEnd,
        (progress) => {
          setProcessingProgress(progress * 100);
        }
      );
      
      setVideoUri(trimmedUri);
      clearEdits();
      
      Alert.alert('Success', 'Video trimmed successfully!');
    } catch (error) {
      Alert.alert('Error', 'Failed to trim video. Please try again.');
      console.error('Trim error:', error);
    } finally {
      setIsProcessing(false);
      setProcessingProgress(0);
    }
  };

  const resetEdits = () => {
    Alert.alert(
      'Reset Edits',
      'Are you sure you want to clear all edits?',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Reset', onPress: clearEdits, style: 'destructive' }
      ]
    );
  };

  return (
    <View style={styles.container}>
      {/* Playback controls */}
      <View style={styles.playbackControls}>
        <TouchableOpacity 
          style={[styles.button, styles.playButton]} 
          onPress={togglePlayback}
        >
          <Text style={styles.buttonText}>
            {isPlaying ? '⏸️' : '▶️'}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Editing actions */}
      <View style={styles.editingControls}>
        <TouchableOpacity 
          style={[styles.button, styles.actionButton]}
          onPress={handleTrimVideo}
          disabled={isProcessing}
        >
          <Text style={styles.buttonText}>
            {isProcessing ? `Trimming ${processingProgress.toFixed(0)}%` : 'Apply Trim'}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={[styles.button, styles.resetButton]}
          onPress={resetEdits}
        >
          <Text style={styles.resetButtonText}>Reset</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 20,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#E5E5E5',
  },
  playbackControls: {
    alignItems: 'center',
    marginBottom: 20,
  },
  editingControls: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  button: {
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  playButton: {
    backgroundColor: '#007AFF',
    width: 60,
    height: 60,
    borderRadius: 30,
  },
  actionButton: {
    backgroundColor: '#34C759',
    flex: 1,
    marginRight: 10,
  },
  resetButton: {
    backgroundColor: '#FF3B30',
    paddingHorizontal: 20,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '500',
  },
  resetButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '500',
  },
});