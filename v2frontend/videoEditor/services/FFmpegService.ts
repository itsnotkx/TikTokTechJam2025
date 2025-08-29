import { FFmpeg } from '@ffmpeg/ffmpeg';
import { fetchFile, toBlobURL } from '@ffmpeg/util';
import * as FileSystem from 'expo-file-system';

class FFmpegService {
  private ffmpeg: FFmpeg | null = null;
  private isLoaded = false;

  async initialize() {
    if (this.isLoaded) return;

    this.ffmpeg = new FFmpeg();
    
    // Load FFmpeg core
    const baseURL = 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd';
    this.ffmpeg.on('log', ({ message }) => {
      console.log(message);
    });

    await this.ffmpeg.load({
      coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, 'text/javascript'),
      wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
    });

    this.isLoaded = true;
  }

  async trimVideo(
    inputUri: string, 
    startTime: number, 
    endTime: number,
    onProgress?: (progress: number) => void
  ): Promise<string> {
    if (!this.ffmpeg || !this.isLoaded) {
      await this.initialize();
    }

    try {
      // Read input file
      const inputData = await fetchFile(inputUri);
      await this.ffmpeg!.writeFile('input.mp4', inputData);

      // Setup progress monitoring
      if (onProgress) {
        this.ffmpeg!.on('progress', ({ progress }) => {
          onProgress(progress);
        });
      }

      // Trim command
      const duration = endTime - startTime;
      await this.ffmpeg!.exec([
        '-i', 'input.mp4',
        '-ss', startTime.toString(),
        '-t', duration.toString(),
        '-c', 'copy', // Use copy to avoid re-encoding for speed
        'output.mp4'
      ]);

      // Get output file
      const outputData = await this.ffmpeg!.readFile('output.mp4');
      
      // Save to device
      const outputUri = `${FileSystem.documentDirectory}trimmed_${Date.now()}.mp4`;
      await FileSystem.writeAsStringAsync(
        outputUri,
        Buffer.from(outputData as Uint8Array).toString('base64'),
        { encoding: FileSystem.EncodingType.Base64 }
      );

      // Cleanup
      await this.ffmpeg!.deleteFile('input.mp4');
      await this.ffmpeg!.deleteFile('output.mp4');

      return outputUri;
    } catch (error) {
      console.error('FFmpeg processing error:', error);
      throw new Error('Video processing failed');
    }
  }

  async applyFilter(
    inputUri: string,
    filterType: 'brightness' | 'contrast' | 'saturation',
    value: number
  ): Promise<string> {
    if (!this.ffmpeg || !this.isLoaded) {
      await this.initialize();
    }

    try {
      const inputData = await fetchFile(inputUri);
      await this.ffmpeg!.writeFile('input.mp4', inputData);

      let filterString = '';
      switch (filterType) {
        case 'brightness':
          filterString = `eq=brightness=${value}`;
          break;
        case 'contrast':
          filterString = `eq=contrast=${value}`;
          break;
        case 'saturation':
          filterString = `eq=saturation=${value}`;
          break;
      }

      await this.ffmpeg!.exec([
        '-i', 'input.mp4',
        '-vf', filterString,
        'output.mp4'
      ]);

      const outputData = await this.ffmpeg!.readFile('output.mp4');
      const outputUri = `${FileSystem.documentDirectory}filtered_${Date.now()}.mp4`;
      
      await FileSystem.writeAsStringAsync(
        outputUri,
        Buffer.from(outputData as Uint8Array).toString('base64'),
        { encoding: FileSystem.EncodingType.Base64 }
      );

      await this.ffmpeg!.deleteFile('input.mp4');
      await this.ffmpeg!.deleteFile('output.mp4');

      return outputUri;
    } catch (error) {
      console.error('Filter processing error:', error);
      throw new Error('Filter application failed');
    }
  }
}

export const ffmpegService = new FFmpegService();