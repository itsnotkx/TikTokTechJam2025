import React from 'react';
import { View, StyleSheet, Dimensions, Text } from 'react-native';
import { PanGestureHandler, GestureHandlerRootView, State } from 'react-native-gesture-handler';
import Animated, {
  useAnimatedGestureHandler,
  useAnimatedStyle,
  useSharedValue,
  runOnJS,
} from 'react-native-reanimated';
import Slider from '@react-native-community/slider';
import { useEditor } from '../../context/EditorContext';

const { width: screenWidth } = Dimensions.get('window');
const TIMELINE_WIDTH = screenWidth - 40;

export const VideoTimeline: React.FC = () => {
  const {
    videoDuration,
    currentTime,
    trimStart,
    trimEnd,
    setCurrentTime,
    setTrimPoints,
  } = useEditor();

  const trimStartX = useSharedValue(0);
  const trimEndX = useSharedValue(TIMELINE_WIDTH);

  // Convert time to X position
  const timeToX = (time: number) => {
    return (time / videoDuration) * TIMELINE_WIDTH;
  };

  // Convert X position to time
  const xToTime = (x: number) => {
    return Math.max(0, Math.min((x / TIMELINE_WIDTH) * videoDuration, videoDuration));
  };

  // Update trim points
  const updateTrimStart = (x: number) => {
    const time = xToTime(x);
    setTrimPoints(time, trimEnd);
  };

  const updateTrimEnd = (x: number) => {
    const time = xToTime(x);
    setTrimPoints(trimStart, time);
  };

  // Gesture handlers for trim handles
  const startGestureHandler = useAnimatedGestureHandler({
    onStart: (_, context) => {
      context.startX = trimStartX.value;
    },
    onActive: (event, context) => {
      const newX = Math.max(0, Math.min(context.startX + event.translationX, trimEndX.value - 20));
      trimStartX.value = newX;
    },
    onEnd: () => {
      runOnJS(updateTrimStart)(trimStartX.value);
    },
  });

  const endGestureHandler = useAnimatedGestureHandler({
    onStart: (_, context) => {
      context.startX = trimEndX.value;
    },
    onActive: (event, context) => {
      const newX = Math.max(trimStartX.value + 20, Math.min(context.startX + event.translationX, TIMELINE_WIDTH));
      trimEndX.value = newX;
    },
    onEnd: () => {
      runOnJS(updateTrimEnd)(trimEndX.value);
    },
  });

  // Animated styles
  const startHandleStyle = useAnimatedStyle(() => ({
    transform: [{ translateX: trimStartX.value }],
  }));

  const endHandleStyle = useAnimatedStyle(() => ({
    transform: [{ translateX: trimEndX.value }],
  }));

  const trimAreaStyle = useAnimatedStyle(() => ({
    left: trimStartX.value,
    width: trimEndX.value - trimStartX.value,
  }));

  // Format time display
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Update trim handle positions when trim points change
  React.useEffect(() => {
    if (videoDuration > 0) {
      trimStartX.value = timeToX(trimStart);
      trimEndX.value = timeToX(trimEnd || videoDuration);
    }
  }, [trimStart, trimEnd, videoDuration]);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Timeline</Text>
      
      {/* Current time scrubber */}
      <View style={styles.scrubberContainer}>
        <Text style={styles.timeLabel}>{formatTime(currentTime)}</Text>
        <Slider
          style={styles.scrubber}
          minimumValue={0}
          maximumValue={videoDuration}
          value={currentTime}
          onValueChange={setCurrentTime}
          minimumTrackTintColor="#007AFF"
          maximumTrackTintColor="#E5E5E5"
          thumbStyle={styles.scrubberThumb}
        />
        <Text style={styles.timeLabel}>{formatTime(videoDuration)}</Text>
      </View>

      {/* Trim interface */}
      <View style={styles.trimContainer}>
        <Text style={styles.trimTitle}>Trim Video</Text>
        
        <View style={styles.trimTimeline}>
          {/* Timeline background */}
          <View style={styles.timelineBackground} />
          
          {/* Selected trim area */}
          <Animated.View style={[styles.trimArea, trimAreaStyle]} />
          
          {/* Start trim handle */}
          <GestureHandlerRootView>
            <PanGestureHandler onGestureEvent={startGestureHandler}>
              <Animated.View style={[styles.trimHandle, styles.startHandle, startHandleStyle]}>
                <View style={styles.handleGrip} />
              </Animated.View>
            </PanGestureHandler>
          </GestureHandlerRootView>
          
          {/* End trim handle */}
          <GestureHandlerRootView>
            <PanGestureHandler onGestureEvent={endGestureHandler}>
              <Animated.View style={[styles.trimHandle, styles.endHandle, endHandleStyle]}>
                <View style={styles.handleGrip} />
              </Animated.View>
            </PanGestureHandler>
          </GestureHandlerRootView>
        </View>
        
        {/* Trim time displays */}
        <View style={styles.trimTimeContainer}>
          <Text style={styles.trimTime}>Start: {formatTime(trimStart)}</Text>
          <Text style={styles.trimTime}>End: {formatTime(trimEnd || videoDuration)}</Text>
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 20,
    backgroundColor: '#f8f9fa',
    borderRadius: 12,
    margin: 10,
  },
  title: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 15,
    color: '#333',
  },
  scrubberContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 30,
  },
  scrubber: {
    flex: 1,
    marginHorizontal: 10,
  },
  scrubberThumb: {
    backgroundColor: '#007AFF',
    width: 20,
    height: 20,
  },
  timeLabel: {
    fontSize: 12,
    color: '#666',
    minWidth: 35,
    textAlign: 'center',
  },
  trimContainer: {
    marginTop: 10,
  },
  trimTitle: {
    fontSize: 16,
    fontWeight: '500',
    marginBottom: 15,
    color: '#333',
  },
  trimTimeline: {
    height: 60,
    position: 'relative',
    marginBottom: 15,
  },
  timelineBackground: {
    position: 'absolute',
    top: 20,
    left: 0,
    right: 0,
    height: 20,
    backgroundColor: '#E5E5E5',
    borderRadius: 10,
  },
  trimArea: {
    position: 'absolute',
    top: 20,
    height: 20,
    backgroundColor: '#007AFF',
    borderRadius: 10,
    opacity: 0.7,
  },
  trimHandle: {
    position: 'absolute',
    top: 10,
    width: 20,
    height: 40,
    backgroundColor: '#007AFF',
    borderRadius: 10,
    justifyContent: 'center',
    alignItems: 'center',
  },
  startHandle: {
    left: 0,
  },
  endHandle: {
    right: 0,
  },
  handleGrip: {
    width: 4,
    height: 16,
    backgroundColor: '#fff',
    borderRadius: 2,
  },
  trimTimeContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  trimTime: {
    fontSize: 12,
    color: '#666',
    backgroundColor: '#fff',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
});